from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

OPENAI_TOOL_FORMAT = "openai_tool"

_OPENAI_ROLES = frozenset({"system", "user", "assistant", "tool", "function", "observation"})
_SHAREGPT_ROLES = frozenset({"system", "human", "gpt", "observation", "function_call"})


def _validate_openai_turn_order(messages: Sequence[Mapping[str, Any]]) -> None:
    """Reject rows that LlamaFactory would otherwise skip with a warning."""

    if messages and messages[0]["role"] == "system":
        messages = messages[1:]
    if any(message["role"] == "system" for message in messages):
        raise ValueError("A system message is only supported as the first message.")
    if not messages:
        raise ValueError("A supervised conversation needs at least one source turn and one assistant target.")

    aligned_roles: list[str] = []
    for message_index, message in enumerate(messages):
        role = message["role"]
        if role == "assistant" and message.get("tool_calls"):
            role = "function"

        if role == "tool":
            if not aligned_roles or aligned_roles[-1] != "function":
                raise ValueError(
                    f"messages[{message_index}] is a tool result but does not follow an assistant tool call."
                )
            # Consecutive tool results are collected into one observation by
            # LlamaFactory's OpenAI converter.
            continue

        if message_index > 0 and message["role"] != "tool" and messages[message_index - 1]["role"] == "tool":
            aligned_roles.append("observation")
        aligned_roles.append(role)

    source_roles = {"user", "observation"}
    target_roles = {"assistant", "function"}
    for turn_index, role in enumerate(aligned_roles):
        expected_roles = source_roles if turn_index % 2 == 0 else target_roles
        if role not in expected_roles:
            expected = "/".join(sorted(expected_roles))
            raise ValueError(
                f"Invalid conversation order at aligned turn {turn_index}: role={role!r}; expected {expected}."
            )
    if len(aligned_roles) % 2 != 0:
        raise ValueError("A supervised conversation must end with an assistant response or tool call.")


def _validate_tool_calls(tool_calls: Any, *, message_index: int) -> None:
    if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, str | bytes) or not tool_calls:
        raise ValueError(f"messages[{message_index}].tool_calls must be a non-empty list.")

    for call_index, call in enumerate(tool_calls):
        if not isinstance(call, Mapping):
            raise ValueError(f"messages[{message_index}].tool_calls[{call_index}] must be an object.")
        function = call.get("function")
        if not isinstance(function, Mapping):
            raise ValueError(f"messages[{message_index}].tool_calls[{call_index}].function must be an object.")
        if not isinstance(function.get("name"), str) or not function["name"].strip():
            raise ValueError(f"messages[{message_index}].tool_calls[{call_index}] has no function name.")
        arguments = function.get("arguments")
        if not isinstance(arguments, str | Mapping):
            raise ValueError(
                f"messages[{message_index}].tool_calls[{call_index}].function.arguments must be JSON text "
                "or an object."
            )


def validate_conversation_record(example: Mapping[str, Any]) -> None:
    """Validate a LlamaFactory conversation record before dataset loading.

    Both native OpenAI-style records (``messages``) and LlamaFactory's
    ShareGPT-style records (``conversations``) are accepted.  The actual
    role alignment and token-level label construction remain LlamaFactory's
    responsibility; this check only produces an early, readable error for
    malformed JSONL rows.
    """

    if not isinstance(example, Mapping):
        raise ValueError("Each conversation JSONL row must be an object.")

    if "messages" in example:
        messages = example["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list.")
        for message_index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise ValueError(f"messages[{message_index}] must be an object.")
            role = message.get("role")
            if role not in _OPENAI_ROLES:
                raise ValueError(f"messages[{message_index}].role={role!r} is not a supported conversation role.")
            if "content" not in message:
                raise ValueError(f"messages[{message_index}] is missing content; use an empty string for tool calls.")
            if message.get("content") is not None:
                if not isinstance(message["content"], str):
                    raise ValueError(f"messages[{message_index}].content must be a string or null.")
            elif role != "assistant" or "tool_calls" not in message:
                raise ValueError(f"messages[{message_index}].content may be null only for an assistant tool call.")
            if "tool_calls" in message:
                if role != "assistant":
                    raise ValueError(f"messages[{message_index}].tool_calls is only valid on assistant messages.")
                if message["content"] not in (None, ""):
                    raise ValueError(
                        f"messages[{message_index}] cannot combine assistant text with tool_calls; "
                        "LlamaFactory serializes that turn as a function call."
                    )
                _validate_tool_calls(message["tool_calls"], message_index=message_index)
        return

    if "conversations" in example:
        conversations = example["conversations"]
        if not isinstance(conversations, list) or not conversations:
            raise ValueError("conversations must be a non-empty list.")
        for message_index, message in enumerate(conversations):
            if not isinstance(message, Mapping):
                raise ValueError(f"conversations[{message_index}] must be an object.")
            role = message.get("from")
            if role not in _SHAREGPT_ROLES:
                raise ValueError(f"conversations[{message_index}].from={role!r} is not supported.")
            if not isinstance(message.get("value"), str):
                raise ValueError(f"conversations[{message_index}].value must be a string.")
        return

    raise ValueError("Conversation rows must contain either a messages or conversations list.")


def normalize_openai_tool_record(example: Mapping[str, Any]) -> dict[str, Any]:
    """Convert standard OpenAI function argument strings into JSON objects.

    The pinned LlamaFactory OpenAI converter expects ``function.arguments``
    to be an object. OpenAI's wire format instead stores it as a JSON string.
    This adapter makes the common OpenAI format safe to train with before
    delegating all role pairing and loss labels to LlamaFactory.
    """

    validate_conversation_record(example)
    if "messages" not in example:
        raise ValueError(f"formatting={OPENAI_TOOL_FORMAT!r} requires a messages column.")

    normalized = dict(example)
    normalized_messages: list[dict[str, Any]] = []
    for message_index, raw_message in enumerate(example["messages"]):
        message = dict(raw_message)
        # The historical OpenAI ``function`` role and some tool datasets'
        # ``observation`` role both denote a tool result. LlamaFactory's
        # OpenAI converter reserves its ``function`` tag for an assistant
        # function-call target, so normalize result roles to ``tool`` first.
        if message["role"] in {"function", "observation"}:
            message["role"] = "tool"
        if "tool_calls" in message:
            normalized_calls: list[dict[str, Any]] = []
            for call_index, raw_call in enumerate(message["tool_calls"]):
                call = dict(raw_call)
                function = dict(call["function"])
                arguments = function["arguments"]
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"messages[{message_index}].tool_calls[{call_index}].function.arguments is not valid JSON."
                        ) from exc
                if not isinstance(arguments, Mapping):
                    raise ValueError(
                        f"messages[{message_index}].tool_calls[{call_index}].function.arguments must decode "
                        "to an object."
                    )
                function["arguments"] = dict(arguments)
                call["function"] = function
                normalized_calls.append(call)
            message["tool_calls"] = normalized_calls
        normalized_messages.append(message)
    normalized["messages"] = normalized_messages
    _validate_openai_turn_order(normalized_messages)
    return normalized


def register_tool_dataset_converters() -> None:
    """Register the OpenAI tool-call adapter with the pinned LlamaFactory.

    ``openai_tool`` has the same top-level shape as OpenAI chat-completion
    data, but fixes the JSON-string ``arguments`` field before LlamaFactory
    renders its native function-call role.  The standard ``sharegpt`` format
    needs no adapter and remains available as-is.
    """

    from llamafactory.data.converter import (
        DATASET_CONVERTERS,
        OpenAIDatasetConverter,
        register_dataset_converter,
    )

    if OPENAI_TOOL_FORMAT in DATASET_CONVERTERS:
        return

    class OpenAIToolDatasetConverter(OpenAIDatasetConverter):
        def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
            return super().__call__(normalize_openai_tool_record(example))

    register_dataset_converter(OPENAI_TOOL_FORMAT, OpenAIToolDatasetConverter)
