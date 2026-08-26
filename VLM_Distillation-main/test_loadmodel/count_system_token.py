import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.arguments import ModelArguments
from src.model.processor import load_processor


def parse_args(model_id):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=model_id)
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant.",
    )
    return parser.parse_args()


def count_system_tokens(processor, system_prompt):
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"][0]
    num_tokens = input_ids.numel()

    print("Rendered system prompt:")
    print(prompt)
    print()
    print(f"Number of system tokens: {num_tokens}")
    print("input_ids:", input_ids.tolist())

    if hasattr(processor, "tokenizer"):
        tokens = processor.tokenizer.convert_ids_to_tokens(input_ids.tolist())
        print("tokens:", tokens)

    return num_tokens


def run_load_test(model_id):
    args = parse_args(model_id)
    model_args = ModelArguments(model_name=args.model_name)

    processor = load_processor(model_args)

    count_system_tokens(
        processor=processor,
        system_prompt=args.system_prompt,
    )


if __name__ == "__main__":
    run_load_test("KamilaMila/FastVLM-0.5B")