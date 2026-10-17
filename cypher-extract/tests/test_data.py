import pytest

from distillation.data import (
    normalize_openai_tool_record,
    validate_conversation_record,
)


def _openai_tool_record(arguments: str = '{"city":"Hà Nội"}') -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Use the available tools."},
            {"role": "user", "content": "Thời tiết Hà Nội thế nào?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_weather",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": arguments},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_weather", "content": '{"temperature_c":31}'},
            {"role": "assistant", "content": "Hà Nội đang 31°C."},
        ],
        "tools": [],
    }


def test_openai_tool_record_normalizes_standard_json_argument_strings() -> None:
    source = _openai_tool_record()
    normalized = normalize_openai_tool_record(source)

    function = normalized["messages"][2]["tool_calls"][0]["function"]
    assert function["name"] == "get_weather"
    assert function["arguments"] == {"city": "Hà Nội"}
    assert source["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"city":"Hà Nội"}'


def test_openai_tool_record_rejects_malformed_argument_json() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        normalize_openai_tool_record(_openai_tool_record("{city: Hà Nội}"))


def test_openai_tool_record_rejects_assistant_text_beside_tool_calls() -> None:
    source = _openai_tool_record()
    source["messages"][2]["content"] = "I will check."
    with pytest.raises(ValueError, match="cannot combine assistant text"):
        normalize_openai_tool_record(source)


def test_openai_tool_record_normalizes_legacy_function_results_to_tool() -> None:
    source = _openai_tool_record()
    source["messages"][3]["role"] = "function"
    normalized = normalize_openai_tool_record(source)
    assert normalized["messages"][3]["role"] == "tool"


def test_openai_tool_record_rejects_tool_result_without_tool_call() -> None:
    source = _openai_tool_record()
    source["messages"][2].pop("tool_calls")
    source["messages"][2]["content"] = "I cannot check that."
    with pytest.raises(ValueError, match="does not follow an assistant tool call"):
        normalize_openai_tool_record(source)


def test_openai_tool_record_rejects_unpaired_assistant_turn() -> None:
    source = _openai_tool_record()
    source["messages"] = source["messages"][:2]
    with pytest.raises(ValueError, match="must end with an assistant response or tool call"):
        normalize_openai_tool_record(source)


def test_openai_tool_record_rejects_system_only_conversation() -> None:
    with pytest.raises(ValueError, match="needs at least one source turn"):
        normalize_openai_tool_record({"messages": [{"role": "system", "content": "Use tools."}]})


def test_openai_tool_record_rejects_non_object_arguments() -> None:
    with pytest.raises(ValueError, match="must decode to an object"):
        normalize_openai_tool_record(_openai_tool_record("[]"))


def test_conversation_validation_accepts_sharegpt_tool_roles() -> None:
    validate_conversation_record(
        {
            "conversations": [
                {"from": "human", "value": "Tìm thời tiết."},
                {"from": "function_call", "value": '{"name":"get_weather","arguments":{"city":"Hà Nội"}}'},
                {"from": "observation", "value": '{"temperature_c":31}'},
                {"from": "gpt", "value": "31°C."},
            ]
        }
    )
