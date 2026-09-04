"""Project-owned VLMEvalKit model adapters for the supported student families."""

from pathlib import Path
import re
import torch
from src.arguments import ModelArguments
from src.model.model import VLMModel
from src.model.processor import load_processor
from vlmeval.vlm.base import BaseModel
from vlmeval.vlm.qwen2_vl.prompt import Qwen2VLPromptMixin
from vlmeval.vlm.qwen3_vl.prompt import Qwen3VLPromptMixin


class _ProjectVLM(BaseModel):
    INTERLEAVE = True

    def __init__(self, model_path, model_backbone, attention_backend="eager", max_new_tokens=128,
                 do_sample=False, use_custom_prompt=False, device_map="auto", **kwargs):
        BaseModel.__init__(self)
        self._use_custom_prompt = bool(use_custom_prompt)
        self.model_path = str(Path(model_path).resolve())
        self.model_backbone = model_backbone
        model_args = ModelArguments(model_name=self.model_path, processor_name=self.model_path,
                                    checkpoint_path=self.model_path, model_backbone=model_backbone)
        load_kwargs = {"attn_implementation": attention_backend}
        if device_map not in {"", "none", None}:
            load_kwargs["device_map"] = device_map
        self.model = VLMModel.load(model_args, is_trainable=False, output_attentions=False, **load_kwargs)
        self.model.eval()
        self.processor = load_processor(model_args)
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)

    def use_custom_prompt(self, dataset):
        return self._use_custom_prompt

    def generation_max_new_tokens(self, dataset=None):
        return self.max_new_tokens

    def generate_inner(self, message, dataset=None):
        content = []
        for item in message:
            if item["type"] == "text":
                content.append({"type": "text", "text": item["value"]})
            elif item["type"] == "image":
                content.append({"type": "image", "image": item["value"]})
            else:
                raise ValueError(f"Unsupported benchmark input type: {item['type']}")
        conversations = [[{"role": "user", "content": content}]]
        inputs = self.processor.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", padding=True)
        device = next(self.model.encoder.parameters()).device
        for key, value in list(inputs.items()):
            if torch.is_tensor(value):
                inputs[key] = value.to(device)
        input_length = inputs["input_ids"].shape[1]
        generated = self.model.generate(**inputs, max_new_tokens=self.generation_max_new_tokens(dataset),
                                        do_sample=self.do_sample, use_cache=True)
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        return tokenizer.batch_decode(generated[:, input_length:], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0].strip()


class ProjectQwen2VLChat(Qwen2VLPromptMixin, _ProjectVLM):
    def __init__(self, model_path, model_backbone, **kwargs):
        if model_backbone not in {"qwen2_vl", "qwen2_5_vl"}:
            raise ValueError(f"Qwen adapter cannot load {model_backbone}")
        super().__init__(model_path=model_path, model_backbone=model_backbone, **kwargs)


class ProjectQwen3VLChat(Qwen3VLPromptMixin, _ProjectVLM):
    """Qwen3-VL adapter with the repository loader and configurable attention."""

    def __init__(self, model_path, model_backbone, **kwargs):
        if model_backbone != "qwen3_vl":
            raise ValueError(f"Qwen3 adapter cannot load {model_backbone}")
        super().__init__(model_path=model_path, model_backbone=model_backbone, **kwargs)


class FastVLMChat(_ProjectVLM):
    """FastVLM adapter using the exact model and processor implementations used for training."""

    def __init__(self, model_path, **kwargs):
        super().__init__(model_path=model_path, model_backbone="fast_vlm", **kwargs)

    @staticmethod
    def _is_mcq(dataset):
        if dataset is None:
            return False
        from vlmeval.dataset import DATASET_TYPE
        return DATASET_TYPE(dataset, default=None) == "MCQ"

    def generation_max_new_tokens(self, dataset=None):
        # This FastVLM checkpoint does not reliably emit EOS for short MCQ
        # answers and otherwise repeats "Answer: X" until the token limit.
        return 1 if self._is_mcq(dataset) else self.max_new_tokens

    def generate_inner(self, message, dataset=None):
        text = super().generate_inner(message, dataset=dataset)
        if self._is_mcq(dataset):
            match = re.match(r"^\s*(?:answer\s*:\s*)?([A-Z])(?:\b|[.):])", text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return text

    def build_prompt(self, line, dataset):
        raise NotImplementedError("FastVLM uses VLMEvalKit's dataset-built prompt")


class OfflineScoringModel(BaseModel):
    """Placeholder used by MODE=eval so existing predictions never reload weights."""

    def __init__(self, **kwargs):
        super().__init__()

    def build_prompt(self, line, dataset):
        raise RuntimeError("OfflineScoringModel cannot perform inference")

    def generate_inner(self, message, dataset=None):
        raise RuntimeError("OfflineScoringModel cannot perform inference")
