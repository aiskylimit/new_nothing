import os
from dataclasses import dataclass, field
from pathlib import Path

from offline_utils import configure_offline_environment, disable_external_reporting
from transformers import AutoTokenizer

from trl import (
    SFTTrainer,
    SFTConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from local_data import load_local_split


configure_offline_environment()


@dataclass
class CustomScriptArguments(ScriptArguments):
    run_config: str = field(default=None, metadata={"help": "Local run name appended to output_dir."})
    local_dataset_path: str = field(
        default=None,
        metadata={"help": "Path to the prepared offline training Dataset."},
    )


def make_format_fn(tokenizer):
    """
    Returns a formatting function that applies the chat template,
    matching the eval prompt format exactly.
    """

    def format_example(example):
        messages = [
            {
                "role": "user",
                "content": f"{example['problem']}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
            },
            {
                "role": "assistant",
                "content": example["solution"],
            },
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}

    return format_example


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    disable_external_reporting(training_args)

    ################
    # Run name and output directory
    ################
    # Extract model name from path (e.g., "Qwen3-1.7B" from "/home/siyanzhao/models/Qwen3-1.7B")
    model_name = model_args.model_name_or_path.split("/")[-1]

    # Format learning rate (e.g., 2e-5 -> "2e-5" or 0.00002 -> "2e-5")
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")

    # Get number of processes from environment (set by accelerate launch)
    num_processes = int(os.environ.get("WORLD_SIZE", 1))

    # Calculate effective batch size
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    # Create concise run name
    run_name = script_args.run_config or f"sft_{model_name}_lr{lr_str}_bs{effective_batch_size}"
    if not training_args.output_dir.endswith(run_name):
        training_args.output_dir = str(Path(training_args.output_dir) / run_name)
    print(f"Run Name: {run_name}")
    print(f"Output Directory: {training_args.output_dir}")

    ################
    # Model & Tokenizer
    ################
    import torch

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "sdpa",
        torch_dtype=torch.bfloat16,
        use_cache=False if training_args.gradient_checkpointing else True,
        local_files_only=True,
    )

    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right",  # Use right padding for SFT
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################

    if not script_args.local_dataset_path:
        raise ValueError("--local_dataset_path is required for offline training.")
    if not Path(model_args.model_name_or_path).expanduser().is_dir():
        raise FileNotFoundError(f"Local model directory does not exist: {model_args.model_name_or_path}")

    train_dataset = load_local_split(script_args.local_dataset_path)
    train_dataset = train_dataset.map(make_format_fn(tokenizer), remove_columns=train_dataset.column_names)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
