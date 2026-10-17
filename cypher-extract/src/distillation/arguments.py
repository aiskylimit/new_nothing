from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, ClassVar

SUPPORTED_METHODS = frozenset(
    {
        "sft",
        "fkl",
        "rkl",
        "sfkl",
        "srkl",
        "bdl",
        "csd",
        "amid",
        "hpd",
        "adaptive_sfkl",
        "adaptive_srkl",
        "fdd_sfkl",
        "fdd_srkl",
    }
)


@dataclass(slots=True)
class DistillationArguments:
    """Arguments specific to the distillation algorithms.

    Model, data, optimization, LoRA, generation, and DeepSpeed options remain
    owned by LlamaFactory. These fields correspond to behavior present in the
    baseline scripts or required by their runtime implementation.
    """

    distill_method: str = "fkl"
    kd_ratio: float = 0.7
    # When set, normalize generator and selector losses independently before
    # mixing them. The generator weight is 1 - selector_loss_weight.
    selector_loss_weight: float | None = None
    # LlamaFactory otherwise copies the student's ``model_revision`` into
    # ``ref_model``, even when teacher and student come from different repos.
    ref_model_revision: str | None = None
    skew_alpha: float = 0.1
    amid_div_name: str = "fkl"
    amid_div_order: str = "pr"
    amid_alpha: float = 0.5
    amid_lam: float = 0.5
    hpd_sample_in_fp32: bool = True
    bdl_lambda: float = 0.9
    # Persist decoded predictions/references and metrics from every generative
    # validation pass under output_dir/eval/ without running generation twice.
    save_eval_predictions: bool = True

    # DistiLLM student rollout controls.
    student_gen: bool = False
    rollout_context_length: int = 810
    gen_top_p: float = 1.0
    init_threshold: float = 0.0
    loss_eps: float = 0.1
    capacity: int = 1000
    replay_ratio: str = "decreasing"

    # FDD controls. Values index Hugging Face `hidden_states`, not transformer
    # block IDs: block ID k maps to hidden-state index k + 1.
    fdd_weight: float | None = None
    student_layer_mapping: list[int] | None = None
    teacher_layer_mapping: list[int] | None = None

    _ALIASES: ClassVar[dict[str, str]] = {
        "adaptive-sfkl": "adaptive_sfkl",
        "adaptive-srkl": "adaptive_srkl",
        "fdd-sfkl": "fdd_sfkl",
        "fdd-srkl": "fdd_srkl",
    }

    def __post_init__(self) -> None:
        self.distill_method = self._ALIASES.get(self.distill_method, self.distill_method)
        if self.student_layer_mapping is None:
            self.student_layer_mapping = []
        if self.teacher_layer_mapping is None:
            self.teacher_layer_mapping = []
        self.validate()

    @property
    def is_adaptive(self) -> bool:
        return self.uses_kd and self.distill_method.startswith("adaptive_")

    @property
    def is_sft(self) -> bool:
        """Whether training should use the student LM loss only.

        ``kd_ratio=0`` is intentionally a first-class SFT mode.  Keeping this
        semantic in the arguments object lets the CLI, workflow, and trainer
        agree that no teacher is needed.
        """

        return self.distill_method == "sft" or self.kd_ratio == 0.0

    @property
    def uses_kd(self) -> bool:
        return not self.is_sft

    @property
    def uses_fdd(self) -> bool:
        return self.uses_kd and self.distill_method.startswith("fdd_")

    @property
    def uses_task_normalized_loss(self) -> bool:
        return self.selector_loss_weight is not None

    @property
    def base_method(self) -> str:
        if self.is_sft:
            return "sft"
        if self.distill_method.startswith("adaptive_"):
            return self.distill_method.removeprefix("adaptive_")
        if self.distill_method.startswith("fdd_"):
            return self.distill_method.removeprefix("fdd_")
        return self.distill_method

    def validate(self) -> None:
        if self.distill_method not in SUPPORTED_METHODS:
            allowed = ", ".join(sorted(SUPPORTED_METHODS))
            raise ValueError(f"Unsupported distill_method={self.distill_method!r}. Expected one of: {allowed}.")
        if not 0.0 <= self.kd_ratio <= 1.0:
            raise ValueError("kd_ratio must be in [0, 1].")
        if self.selector_loss_weight is not None and not 0.0 < self.selector_loss_weight < 1.0:
            raise ValueError("selector_loss_weight must be in (0, 1) when task-normalized loss is enabled.")
        if self.distill_method == "sft" and self.kd_ratio != 0.0:
            raise ValueError("distill_method=sft requires kd_ratio=0.")
        if self.base_method in {"sfkl", "srkl"} and not 0.0 < self.skew_alpha < 1.0:
            raise ValueError("skew_alpha must be in (0, 1).")
        if self.amid_div_name not in {"fkl", "ab"}:
            raise ValueError("amid_div_name must be fkl or ab.")
        if self.amid_div_order not in {"pr", "qr", "rp", "rq"}:
            raise ValueError("amid_div_order must be one of pr, qr, rp, or rq.")
        if not 0.0 <= self.amid_lam <= 1.0:
            raise ValueError("amid_lam must be in [0, 1].")
        if not 0.0 < self.bdl_lambda < 1.0:
            raise ValueError("bdl_lambda must be in (0, 1).")
        if math.isclose(self.bdl_lambda, 0.5, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("bdl_lambda=0.5 makes BDL identically zero.")
        if not 0.0 < self.gen_top_p <= 1.0:
            raise ValueError("gen_top_p must be in (0, 1].")
        if not 0.0 <= self.init_threshold <= 1.0:
            raise ValueError("init_threshold must be in [0, 1].")
        if self.loss_eps < 0.0:
            raise ValueError("loss_eps must be non-negative.")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive.")
        if self.replay_ratio not in {"constant", "increasing", "decreasing"}:
            raise ValueError("replay_ratio must be constant, increasing, or decreasing.")

        # A zero KD ratio is an explicit request for SFT, even when the YAML
        # was copied from an adaptive or FDD config and still contains its
        # method-specific knobs.  Those knobs are inert in the SFT path.
        if self.is_sft:
            return

        if self.is_adaptive and not self.student_gen:
            raise ValueError(f"{self.distill_method} is on-policy and requires student_gen=true.")
        if not self.is_adaptive and self.student_gen:
            raise ValueError("student_gen is only valid for adaptive DistiLLM methods.")
        if self.is_adaptive and self.rollout_context_length <= 0:
            raise ValueError("rollout_context_length must be positive.")

        if self.uses_fdd:
            if self.fdd_weight is None or not 0.0 <= self.fdd_weight <= 1.0:
                raise ValueError("FDD requires fdd_weight in [0, 1].")
            if len(self.student_layer_mapping) != len(self.teacher_layer_mapping):
                raise ValueError("FDD student and teacher layer mappings must have equal length.")
            if len(self.student_layer_mapping) < 2:
                raise ValueError("FDD requires at least two student/teacher layer pairs.")
        elif self.student_layer_mapping or self.teacher_layer_mapping or self.fdd_weight is not None:
            raise ValueError("FDD arguments can only be used with an fdd_* method.")

    def validate_runtime(self, *, cutoff_len: int, per_device_train_batch_size: int) -> None:
        if self.is_adaptive and self.rollout_context_length >= cutoff_len:
            raise ValueError("rollout_context_length must be smaller than LlamaFactory cutoff_len.")
        if self.is_adaptive and self.capacity < per_device_train_batch_size:
            raise ValueError("DistiLLM capacity must hold at least one per-device training batch.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def field_names(cls) -> frozenset[str]:
        return frozenset(item.name for item in fields(cls))

    @classmethod
    def split_config(cls, config: dict[str, Any]) -> tuple[DistillationArguments, dict[str, Any]]:
        custom_names = cls.field_names()
        custom = {key: value for key, value in config.items() if key in custom_names}
        remaining = {key: value for key, value in config.items() if key not in custom_names}
        return cls(**custom), remaining
