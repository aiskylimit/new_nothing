# Train multi-turn tool calls

The repository trains native LlamaFactory multi-turn tool-use conversations.
Every shipped config already selects the appropriate model-native template:

| Model family | Template |
| --- | --- |
| Llama 3.x | `llama3` |
| Qwen 3 without synthetic reasoning spans | `qwen3_nothink` |

The supplied [`sft.yaml`](../configs/qwen3/sft.yaml) configs make these template
choices. They omit `mask_history` because the Cypher datasets are single-turn.
For a multi-turn tool dataset, keep `mask_history: false` (the LlamaFactory
default) so every assistant/function-call turn receives supervision;
`mask_history: true` would keep only the final assistant turn.

## Recommended OpenAI-style JSONL

Use `formatting: openai_tool` in `data/dataset_info.json`. This repository
registers that adapter before LlamaFactory loads data. It accepts ordinary
OpenAI `tool_calls`, including JSON *strings* in `function.arguments`, and
normalizes them before LlamaFactory renders the model-specific tool syntax.
For a tool-call turn, keep assistant `content` empty or `null`: this LlamaFactory
format serializes that turn as the function call itself. Historical OpenAI
`function` result roles are accepted and normalized to the current `tool` role.
Rows must alternate source turns (`user` or tool result) and learned turns
(assistant text or assistant tool call); malformed rows fail during preprocessing
instead of being silently skipped by LlamaFactory.

```json
{
  "messages": [
    {"role": "system", "content": "You can use the supplied tools."},
    {"role": "user", "content": "Thời tiết Hà Nội thế nào?"},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_weather",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"city\":\"Hà Nội\"}"
        }
      }]
    },
    {
      "role": "tool",
      "tool_call_id": "call_weather",
      "content": "{\"temperature_c\":31,\"condition\":\"rain\"}"
    },
    {"role": "assistant", "content": "Hà Nội đang 31°C và có mưa."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }
  ]
}
```

Register the train/validation files as follows. The `tags` block is required
because the raw record uses OpenAI's `role`/`content` keys rather than
ShareGPT's `from`/`value` keys.

```json
{
  "tool_train": {
    "file_name": "tool_data/train.jsonl",
    "formatting": "openai_tool",
    "columns": {"messages": "messages", "tools": "tools"},
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "observation_tag": "tool",
      "function_tag": "function",
      "system_tag": "system"
    }
  }
}
```

`openai_tool` turns the previous trajectory into LlamaFactory's role sequence:

```text
user → function_call → observation → assistant
```

With `train_on_prompt: false` and `mask_history: false`, the resulting labels
are:

```text
system/user/tool observation     -> -100 (context only)
assistant tool call              -> target tokens
final assistant answer           -> target tokens
```

Therefore both the decision to call a function and its serialized arguments
receive CE/KD supervision, but a tool response is never a target. For native
conversation templates, this repository also applies the same target mask to
the FDD feature-loss branch.

## Run the included smoke example

`data/tool_use_demo.jsonl` is a one-record OpenAI-style example registered as
`tool_use_demo`.

```bash
uv run bash scripts/train.sh configs/qwen3/sft.yaml \
  dataset=tool_use_demo eval_dataset=tool_use_demo
```

Use `configs/llama3/sft.yaml` for Llama 3. All static KD configs already use
the native template, for example:

```bash
uv run bash scripts/train.sh configs/qwen3/fkl.yaml \
  dataset=tool_train eval_dataset=tool_validation \
  ref_model_adapters=/path/to/teacher-adapter
```

Adaptive DistiLLM supports tool trajectories by extracting every contiguous
supervised assistant span as an independent rollout prompt. This trains both
function-call and final-answer behavior. For a final-answer rollout, the gold
tool observation stays in context; adaptive generation does not execute tools
or replace that observation with a live environment response.

## ShareGPT alternative

Plain LlamaFactory `sharegpt` data also works without the adapter. Use
alternating roles `human`/`observation` and `gpt`/`function_call`:

```json
{
  "conversations": [
    {"from": "human", "value": "Thời tiết Hà Nội thế nào?"},
    {"from": "function_call", "value": "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Hà Nội\"}}"},
    {"from": "observation", "value": "{\"temperature_c\":31}"},
    {"from": "gpt", "value": "Hà Nội đang 31°C."}
  ],
  "tools": "[{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\"}}]"
}
```

For ShareGPT, `tools` must be a JSON string, and `function_call.value` must
contain an object-valued `arguments` field (not an OpenAI JSON argument
string). The OpenAI-style form above is usually less error-prone.
