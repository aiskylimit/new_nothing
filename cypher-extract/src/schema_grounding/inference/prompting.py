from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

Message = dict[str, str]

_QWEN3_NOTHINK_MESSAGE = "<|im_start|>{role}\n{content}<|im_end|>\n"
_QWEN3_NOTHINK_GENERATION_PROMPT = "<|im_start|>assistant\n"
_LLAMA3_MESSAGE = "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
_LLAMA3_GENERATION_PROMPT = "<|start_header_id|>assistant<|end_header_id|>\n\n"
QWEN3_NOTHINK_TEMPLATE_NAME = "llamafactory:qwen3_nothink"
QWEN2_5_TEMPLATE_NAME = "llamafactory:qwen"
QWEN3_NOTHINK_TEMPLATE_FINGERPRINT = sha256(
    f"{_QWEN3_NOTHINK_MESSAGE}\0{_QWEN3_NOTHINK_GENERATION_PROMPT}".encode()
).hexdigest()
QWEN2_5_TEMPLATE_FINGERPRINT = QWEN3_NOTHINK_TEMPLATE_FINGERPRINT
LLAMA3_TEMPLATE_NAME = "llamafactory:llama3"
LLAMA3_TEMPLATE_FINGERPRINT = sha256(
    f"{_LLAMA3_MESSAGE}\0{_LLAMA3_GENERATION_PROMPT}".encode()
).hexdigest()


def qwen_template_metadata(model_family: str) -> dict[str, str]:
    if model_family == "qwen2.5_coder":
        return {"name": QWEN2_5_TEMPLATE_NAME, "fingerprint": QWEN2_5_TEMPLATE_FINGERPRINT}
    return {
        "name": QWEN3_NOTHINK_TEMPLATE_NAME,
        "fingerprint": QWEN3_NOTHINK_TEMPLATE_FINGERPRINT,
    }


def chat_template_metadata(model_family: str) -> dict[str, str]:
    if model_family == "llama3":
        return {"name": LLAMA3_TEMPLATE_NAME, "fingerprint": LLAMA3_TEMPLATE_FINGERPRINT}
    return qwen_template_metadata(model_family)


def render_qwen3_nothink(
    messages: list[Message],
    *,
    add_generation_prompt: bool = True,
) -> str:
    """Render the exact non-reasoning ChatML format used by LlamaFactory."""

    rendered: list[str] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported qwen3_nothink message role: {role!r}")
        if not isinstance(content, str):
            raise TypeError("qwen3_nothink message content must be a string")
        rendered.append(_QWEN3_NOTHINK_MESSAGE.format(role=role, content=content))
    if add_generation_prompt:
        rendered.append(_QWEN3_NOTHINK_GENERATION_PROMPT)
    return "".join(rendered)


def render_llama3(
    messages: list[Message],
    *,
    bos_token: str,
    add_generation_prompt: bool = True,
) -> str:
    """Render the exact LlamaFactory llama3 format without native date injection."""

    rendered = [bos_token]
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported llama3 message role: {role!r}")
        if not isinstance(content, str):
            raise TypeError("llama3 message content must be a string")
        rendered.append(_LLAMA3_MESSAGE.format(role=role, content=content))
    if add_generation_prompt:
        rendered.append(_LLAMA3_GENERATION_PROMPT)
    return "".join(rendered)


@dataclass(frozen=True)
class PromptTemplates:
    generator_system: str
    generator_user: str
    selector_system: str
    selector_user: str

    def fingerprints(self) -> dict[str, str]:
        """Return stable content hashes used to validate resumable outputs."""

        return {
            name: sha256(getattr(self, name).encode("utf-8")).hexdigest()
            for name in (
                "generator_system",
                "generator_user",
                "selector_system",
                "selector_user",
            )
        }

    @classmethod
    def from_repository(cls, repository_root: Path) -> PromptTemplates:
        prompt_root = repository_root / "prompts"

        def read(relative_path: str) -> str:
            return (prompt_root / relative_path).read_text(encoding="utf-8").strip()

        return cls(
            generator_system=read("generator/system_prompt.txt"),
            generator_user=read("generator/user_prompt.txt"),
            selector_system=read("selector/system_prompt.txt"),
            selector_user=read("selector/user_prompt.txt"),
        )

    def selector_messages(self, question: str, schema_unit: str) -> list[Message]:
        return [
            {"role": "system", "content": self.selector_system},
            {
                "role": "user",
                "content": self.selector_user.format(question=question, schema_unit=schema_unit),
            },
        ]

    def generator_messages(self, question: str, sub_schema: dict[str, Any]) -> list[Message]:
        schema = json.dumps(sub_schema, ensure_ascii=False, indent=2)
        return [
            {"role": "system", "content": self.generator_system},
            {
                "role": "user",
                "content": self.generator_user.format(question=question, schema=schema),
            },
        ]
