import json
import os

import torch
import torch.nn as nn

from src.arguments import ModelArguments, TrainingArguments
from src.model.model import VLMModel
from src.model.processor import load_processor
from src.utils import print_master, print_rank


def _create_teacher_model_args(model_args: ModelArguments) -> ModelArguments:
    return ModelArguments(
        model_name=model_args.teacher_model_name,
        model_backbone=model_args.teacher_backbone,
        lora=model_args.teacher_lora,
        lora_r=model_args.teacher_lora_r,
        lora_alpha=model_args.teacher_lora_alpha,
        lora_dropout=model_args.teacher_lora_dropout,
        lora_target_modules=model_args.teacher_lora_target_modules,
        pooling=model_args.teacher_pooling,
        normalize=model_args.teacher_normalize,
    )


def _init_semi_orthogonal(tensor: torch.Tensor) -> torch.Tensor:
    rows, cols = tensor.shape
    if rows >= cols:
        a = torch.randn(rows, cols, dtype=tensor.dtype)
        q, _ = torch.linalg.qr(a, mode="reduced")
        tensor.data[:] = q[:, :cols]
    else:
        a = torch.randn(cols, rows, dtype=tensor.dtype)
        q, _ = torch.linalg.qr(a, mode="reduced")
        tensor.data[:] = q.T[:rows, :]
    return tensor


def _resolve_dskd_topk_vocab(args) -> int:
    value = int(getattr(args, "dskd_topk_vocab", -1) or -1)
    if value == -1:
        value = int(getattr(args, "topk_vocab", -1) or -1)
    return value


class Distiller(nn.Module):
    def __init__(self, model_args: ModelArguments, training_args: TrainingArguments):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args

        self.student = self._load_student()
        self.teacher = self._load_teacher()

        self.student_hidden_dim = model_args.student_hidden_dim
        self.teacher_hidden_dim = model_args.teacher_hidden_dim

        self.set_projector()
        self.load_projectors_if_needed()
        self.init_dskd_projectors_if_needed()
        print_master("Projectors set.")

    @property
    def config(self):
        return getattr(self.student, "config", None) or getattr(getattr(self.student, "encoder", None), "config", None)

    @property
    def generation_config(self):
        encoder = getattr(self.student, "encoder", None)
        return getattr(encoder, "generation_config", None) if encoder is not None else None

    @property
    def name_or_path(self):
        return getattr(self.model_args, "model_name", "") or ""

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _needs_attention_outputs(self) -> bool:
        kd_loss_type = (getattr(self.training_args, "kd_loss_type", "") or "").lower()
        attention_free_kd_losses = {
            "ce_only",
            "default",
            "default_distillation",
            "emkd",
            "em_kd",
            "dwa_kd",
            "dwakd",
            "dskd_v2",
            "dskdv2",
            "dskd_v2_with_eta",
            "dskdv2_with_eta",
            "mcw_kd",
            "mcwkd",
            "cgkd",
        }
        # Unknown/future criteria keep the conservative old behaviour.  SRE,
        # SCVA, and their joint criteria consume attention matrices directly.
        return kd_loss_type not in attention_free_kd_losses

    def _load_student(self) -> VLMModel:
        print_master(
            f"Loading student: {self.model_args.model_name} "
            f"(lora={self.model_args.lora}, r={self.model_args.lora_r})"
        )
        student = VLMModel.build(
            self.model_args,
            output_attentions=self._needs_attention_outputs(),
        )
        print_master("Student model built.")
        return student

    def _load_teacher(self) -> VLMModel:
        teacher_model_args = _create_teacher_model_args(self.model_args)
        print_master(
            f"Loading teacher: {teacher_model_args.model_name} "
            f"(lora={teacher_model_args.lora}, r={teacher_model_args.lora_r})"
        )
        teacher = VLMModel.load(
            teacher_model_args,
            is_trainable=False,
            output_attentions=self._needs_attention_outputs(),
        )
        for param in teacher.parameters():
            param.requires_grad = False
        teacher.eval()
        print_master("Teacher model loaded.")
        return teacher

    # ------------------------------------------------------------------
    # Processor helpers
    # ------------------------------------------------------------------

    def get_student_processor(self):
        if hasattr(self, "_student_processor"):
            return self._student_processor
        processor = load_processor(self.model_args)
        self._student_processor = processor
        print_master("Student processor loaded.")
        return processor

    def get_teacher_processor(self):
        if hasattr(self, "_teacher_processor"):
            return self._teacher_processor
        teacher_model_args = _create_teacher_model_args(self.model_args)
        processor = load_processor(teacher_model_args)
        self._teacher_processor = processor
        print_master("Teacher processor loaded.")
        return processor

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, criterion, batch):
        return criterion(self, batch)

    # ------------------------------------------------------------------
    # Projector
    # ------------------------------------------------------------------

    def set_projector(self):
        kd_loss_type = (getattr(self.training_args, "kd_loss_type", "") or "").lower()
        if kd_loss_type in {"dskd_v2_with_eta", "dskdv2_with_eta", "dskd_v2", "dskdv2"}:
            # DSKDv2 owns a fixed pair of dual-space mappings.  Their shapes
            # follow directly from the two LM hidden dimensions, and
            # init_dskd_projectors_if_needed() initializes/caches the mappings
            # from the LM-head pseudo-inverses.  No external JSON is required.
            self.projectors = nn.ModuleDict(
                {
                    "t2s": nn.Sequential(
                        nn.Linear(self.teacher_hidden_dim, self.student_hidden_dim, dtype=torch.bfloat16)
                    ),
                    "s2t": nn.Sequential(
                        nn.Linear(self.student_hidden_dim, self.teacher_hidden_dim, dtype=torch.bfloat16)
                    ),
                }
            )
        elif self.model_args.projector_config_path is not None:
            self.projectors = nn.ModuleDict()
            with open(self.model_args.projector_config_path) as f:
                projector_config = json.load(f)

            dim_map = {
                "s": self.student_hidden_dim,
                "t": self.teacher_hidden_dim,
                "p": self.model_args.proj_dim,
            }

            for name, cfg in projector_config.items():
                if not cfg.get("enabled", False):
                    continue

                seq = nn.Sequential()
                parts = cfg["structure"].split("-")
                parsed = []

                for p in parts:
                    if p == "relu":
                        parsed.append("relu")
                    elif p.isdigit():
                        parsed.append(int(p))
                    else:
                        suffix = p[-1]
                        coef = int(p[:-1]) if len(p) > 1 and p[:-1].isdigit() else 1
                        parsed.append(coef * dim_map[suffix])

                for i in range(len(parsed) - 1):
                    a, b = parsed[i], parsed[i + 1]
                    if isinstance(a, int) and isinstance(b, int):
                        layer = nn.Linear(a, b)
                        _init_semi_orthogonal(layer.weight)
                        seq.append(layer.to(dtype=torch.bfloat16))
                    elif b == "relu":
                        seq.append(nn.ReLU())
                    elif a == "relu" and isinstance(b, int):
                        prev = parsed[i - 1] if i > 0 and isinstance(parsed[i - 1], int) else None
                        if prev is not None:
                            layer = nn.Linear(prev, b)
                            _init_semi_orthogonal(layer.weight)
                            seq.append(layer.to(dtype=torch.bfloat16))

                self.projectors[name] = seq
        else:
            projector_list = nn.ModuleList()
            for _ in range(len(self.training_args.teacher_layer_mapping)):
                projector_list.append(
                    nn.Linear(self.student_hidden_dim, self.teacher_hidden_dim, dtype=torch.bfloat16)
                )
            self.projectors = projector_list

        print_master(f"Created {len(self.projectors)} projector(s).")

    def add_optimizer_param_group(self, optimizer):
        if hasattr(self, "projectors") and self.projectors is not None:
            lr = self.model_args.projector_lr or self.training_args.learning_rate
            optimizer.add_param_group({"params": self.projectors.parameters(), "lr": lr})
            print_master("Projector parameters added to optimizer.")
        return optimizer

    def save_projectors(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        if isinstance(self.projectors, nn.ModuleDict):
            for name, proj in self.projectors.items():
                path = os.path.join(save_dir, f"projector_{name}.pt")
                torch.save(proj.state_dict(), path)
                print_master(f"Projector '{name}' saved to {path}")
        else:
            for i, proj in enumerate(self.projectors):
                path = os.path.join(save_dir, f"projector_{i}.pt")
                torch.save(proj.state_dict(), path)
                print_master(f"Projector {i} saved to {path}")

    def load_projectors_if_needed(self):
        if not self.model_args.projector_path:
            return

        path = self.model_args.projector_path
        if isinstance(self.projectors, nn.ModuleDict):
            if os.path.isfile(path):
                state = torch.load(path, map_location="cpu")
                self.projectors.load_state_dict(state)
                print_master(f"Projectors loaded from {path}")
                return

            for name, proj in self.projectors.items():
                proj_path = os.path.join(path, f"projector_{name}.pt")
                if os.path.exists(proj_path):
                    proj.load_state_dict(torch.load(proj_path, map_location="cpu"))
                    print_master(f"Projector '{name}' loaded from {proj_path}")
            return

        if os.path.isfile(path):
            state = torch.load(path, map_location="cpu")
            self.projectors.load_state_dict(state)
            print_master(f"Projectors loaded from {path}")
            return

        for i, proj in enumerate(self.projectors):
            proj_path = os.path.join(path, f"projector_{i}.pt")
            if os.path.exists(proj_path):
                proj.load_state_dict(torch.load(proj_path, map_location="cpu"))
                print_master(f"Projector {i} loaded from {proj_path}")

    def init_dskd_projectors_if_needed(self):
        kd_loss_type = (getattr(self.training_args, "kd_loss_type", "") or "").lower()
        if kd_loss_type not in {"dskd_v2_with_eta", "dskdv2_with_eta", "dskd_v2", "dskdv2"}:
            return
        if not (getattr(self.training_args, "init_t2s_projector", False) or getattr(self.training_args, "init_s2t_projector", False)):
            return
        if not isinstance(self.projectors, nn.ModuleDict):
            raise RuntimeError("DSKDv2 internal projector creation did not produce a named ModuleDict.")
        if "t2s" not in self.projectors or "s2t" not in self.projectors:
            raise RuntimeError("DSKDv2 internal projector creation must provide `t2s` and `s2t`.")

        student_head = self._output_head_weight(self.student).detach().transpose(0, 1).float()
        teacher_head = self._output_head_weight(self.teacher).detach().transpose(0, 1).float()
        student_tokenizer = getattr(self.get_student_processor(), "tokenizer", self.get_student_processor())
        teacher_tokenizer = getattr(self.get_teacher_processor(), "tokenizer", self.get_teacher_processor())

        if student_tokenizer.get_vocab().items() == teacher_tokenizer.get_vocab().items():
            part_student_head = student_head
            part_teacher_head = teacher_head
            self.student_overlap_token_ids = None
        else:
            student_vocab = {token.replace("Ġ", "▁"): idx for token, idx in student_tokenizer.get_vocab().items()}
            teacher_vocab = {token.replace("Ġ", "▁"): idx for token, idx in teacher_tokenizer.get_vocab().items()}
            overlap_tokens = [token for token in student_vocab if token in teacher_vocab]
            if not overlap_tokens:
                raise RuntimeError("DSKDv2 projector initialization found no overlap tokens between student and teacher.")
            student_overlap_token_ids = torch.tensor([student_vocab[token] for token in overlap_tokens], dtype=torch.long, device=student_head.device)
            teacher_overlap_token_ids = torch.tensor([teacher_vocab[token] for token in overlap_tokens], dtype=torch.long, device=teacher_head.device)
            part_student_head = student_head.index_select(1, student_overlap_token_ids)
            part_teacher_head = teacher_head.index_select(1, teacher_overlap_token_ids)
            self.student_overlap_token_ids = student_overlap_token_ids.cpu()
            print_master(f"DSKDv2 projector init overlap tokens: {len(overlap_tokens)}")

        topk_vocab = _resolve_dskd_topk_vocab(self.training_args)
        if topk_vocab != -1:
            part_student_head = part_student_head[:, :topk_vocab]
            part_teacher_head = part_teacher_head[:, :topk_vocab]
            if getattr(self, "student_overlap_token_ids", None) is not None:
                self.student_overlap_token_ids = self.student_overlap_token_ids[:topk_vocab]

        if getattr(self.training_args, "init_t2s_projector", False):
            part_student_head_pinv = torch.linalg.pinv(part_student_head)
            init_t2s = (part_teacher_head @ part_student_head_pinv).transpose(0, 1)
            self._copy_linear_weight(self.projectors["t2s"], init_t2s)
            print_master("DSKDv2 initialized t2s projector with LM-head pseudo-inverse.")

        if getattr(self.training_args, "init_s2t_projector", False):
            pinv_dtype = next(self.projectors["s2t"].parameters()).dtype
            self.part_teacher_head_pinv = torch.linalg.pinv(part_teacher_head).to(dtype=pinv_dtype).detach()
            # The dynamic branch derives s2t from the frozen LM heads on every
            # forward and never calls this configured module.  Keeping it
            # trainable would allocate optimizer state for unused parameters.
            self.projectors["s2t"].requires_grad_(False)
            print_master("DSKDv2 cached teacher-head pseudo-inverse for dynamic s2t projection.")

    @staticmethod
    def _output_head_weight(model: nn.Module) -> torch.Tensor:
        encoder = getattr(model, "encoder", model)
        if hasattr(encoder, "get_output_embeddings"):
            head = encoder.get_output_embeddings()
            if head is not None:
                return head.weight
        if hasattr(encoder, "lm_head"):
            return encoder.lm_head.weight
        raise RuntimeError(f"Could not find output head for model: {type(model)}")

    @staticmethod
    def _copy_linear_weight(projector: nn.Module, weight: torch.Tensor) -> None:
        linear = None
        if isinstance(projector, nn.Linear):
            linear = projector
        elif isinstance(projector, nn.Sequential):
            for module in projector:
                if isinstance(module, nn.Linear):
                    linear = module
                    break
        if linear is None:
            raise RuntimeError("Expected projector to contain an nn.Linear for DSKDv2 initialization.")
        if tuple(linear.weight.shape) != tuple(weight.shape):
            raise RuntimeError(f"Projector weight shape {tuple(linear.weight.shape)} does not match init {tuple(weight.shape)}.")
        with torch.no_grad():
            linear.weight.copy_(weight.to(device=linear.weight.device, dtype=linear.weight.dtype))
            if linear.bias is not None:
                linear.bias.zero_()
