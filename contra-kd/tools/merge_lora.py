from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base-model", type=str)
parser.add_argument("--lora", type=str)
parser.add_argument("--output-dir", type=str, required=True)
args = parser.parse_args()
BASE_MODEL_ID = args.base_model
LORA_PATH = args.lora # local directory containing adapter_config.json

# Load base model
config = AutoConfig.from_pretrained(BASE_MODEL_ID, local_files_only=True)
config.is_model_parallel = False

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, config=config, device_map="cpu", torch_dtype=torch.float16, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, local_files_only=True)

# this is only for Qwen models, adjust when using
# tokenizer.pad_token_id = 151646
# tokenizer.eos_token_id = 151643

tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'  # Set left padding for decoder-only models
print("Tokenizer size:", len(tokenizer))

# (Only needed if vocab was resized during LoRA training)
model.resize_token_embeddings(len(tokenizer))

model.enable_input_require_grads()

model = PeftModel.from_pretrained(model, LORA_PATH, local_files_only=True)

# Merge LoRA weights into base model and unload PEFT
model = model.merge_and_unload()

# Optional but recommended for inference export
model.eval()

# Persist both artifacts on the local filesystem only.
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
