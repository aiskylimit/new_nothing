"""Privileged prompt formatting; supporting context is visible only to the teacher."""

from .records import get_raw_prompt

DEFAULT_CONTEXT_TEMPLATE = "\n\nAdditional context:\n{privileged_context}\n\n"


def render_privileged_teacher_prompt(record, tokenizer, context, template):
    """Shared rendering for preparation checks and training (empty context allowed)."""
    if "{privileged_context}" not in template:
        raise ValueError("Context template must contain {privileged_context}")
    addition = template.format(privileged_context=context)
    # Never append context after a chat assistant generation header.
    if "privileged_prompt" in record:
        prompt = record["privileged_prompt"]
        if not isinstance(prompt, str) or "{privileged_context}" not in prompt:
            raise ValueError("privileged_prompt must contain {privileged_context}")
        prompt = prompt.replace("{privileged_context}", addition)
    else:
        user_prompt = get_raw_prompt(record, tokenizer)
        messages = []
        if record.get("system_prompt"):
            messages.append({"role": "system", "content": record["system_prompt"]})
        messages.append({"role": "user", "content": user_prompt + addition})
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        ) if getattr(tokenizer, "chat_template", None) else "\n\n".join(m["content"] for m in messages)
    return prompt


def build_privileged_teacher_input(record, tokenizer, field, template):
    """Render question and context; batch packing applies teacher training limits."""
    context = record.get(field)
    if not isinstance(context, str) or not context.strip():
        raise ValueError(f"Privileged training requires a nonempty '{field}' string")
    prompt = render_privileged_teacher_prompt(record, tokenizer, context, template)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids:
        raise ValueError("Privileged prompt must contain at least one token")
    return ids, len(tokenizer.encode(context, add_special_tokens=False))
