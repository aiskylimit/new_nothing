"""Đường inference cho Ours (spec §7.4 — chỗ proposal bỏ ngỏ).

Ba chế độ, tất cả đều dựa trên một quan sát: LoRA là cộng tính, nên trộn expert
theo trọng số ``g`` chỉ là đặt lại ``scaling`` của từng adapter:

    W_eff = W_base + sum_m g_m * (alpha/r) * B_m A_m

Không cần merge tensor, chỉ ghi lại vài trăm số float -> đổi route giữa chừng
lúc generate gần như miễn phí.

    merged      : g = ḡ (trung bình gate trên tập train). Một adapter tĩnh,
                  chạy được bằng vLLM -> dùng cho bảng chính.
    per_example : g tính từ hidden state cuối prompt, cố định trong cả lời giải.
    dynamic     : g tính lại ở mỗi ranh giới step ("\\n\\n") trong lúc generate.
                  Trung thành nhất với training objective, nhưng phải chạy bằng
                  HF generate loop -> chậm, chỉ dùng cho subset / ablation.

Lý do chọn mặc định và caveat: xem ``INFERENCE.md``.
"""

from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from core.gate import StepGate
from core.model_utils import (
    expert_names,
    load_base_model,
    load_experts,
    load_tokenizer,
    set_active_adapters,
)
from core.segmentation import render_prompt


# --------------------------------------------------------------------------- #
# Trộn expert bằng cách đặt lại scaling
# --------------------------------------------------------------------------- #
class AdapterMixer:
    """Đặt ``scaling[adapter] = base_scaling * g_m`` cho mọi LoRA layer."""

    def __init__(self, model, adapter_names: Sequence[str]):
        self.model = model
        self.names = list(adapter_names)
        self.layers = []
        for _, module in model.named_modules():
            scaling = getattr(module, "scaling", None)
            if isinstance(scaling, dict) and all(n in scaling for n in self.names):
                base = {n: float(scaling[n]) for n in self.names}
                self.layers.append((module, base))
        if not self.layers:
            raise RuntimeError("Không tìm thấy LoRA layer nào mang đủ M adapter")
        set_active_adapters(model, self.names)  # bật đồng thời cả M adapter

    def set_mixture(self, g: Sequence[float]) -> None:
        if len(g) != len(self.names):
            raise ValueError(f"g có {len(g)} phần tử, cần {len(self.names)}")
        for module, base in self.layers:
            for name, weight in zip(self.names, g):
                # Ưu tiên API công khai của PEFT nếu bản đang cài có; nếu không
                # thì ghi thẳng vào dict (chi tiết nội bộ, có thể đổi theo bản).
                setter = getattr(module, "set_scale", None)
                if callable(setter):
                    setter(name, float(weight))
                else:
                    module.scaling[name] = base[name] * float(weight)

    def reset(self) -> None:
        for module, base in self.layers:
            for name in self.names:
                module.scaling[name] = base[name]


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
@dataclass
class SamplingConfig:
    temperature: float = 0.6
    top_p: float = 0.95
    max_new_tokens: int = 32768
    seed: int = 42


class GateRouter:
    """Base + M expert + gate đã train, phục vụ 3 chế độ inference."""

    def __init__(
        self,
        model_name: str,
        phase1_dir: str,
        gate_dir: str,
        *,
        n_experts: int = 3,
        device: Optional[torch.device] = None,
        step_separator: str = "\n\n",
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available()
                                 else torch.device("cpu"))
        self.tokenizer = load_tokenizer(model_name)
        names = expert_names(n_experts)
        base = load_base_model(model_name).to(self.device)
        self.model, self.names = load_experts(
            base, [os.path.join(phase1_dir, n) for n in names], names
        )
        self.model.eval()
        self.model.config.use_cache = True  # load_base_model tắt cache cho train
        self.mixer = AdapterMixer(self.model, self.names)
        self.gate = StepGate.load(gate_dir).to(self.device).eval()
        with open(os.path.join(gate_dir, "gate_config.json"), "r", encoding="utf-8") as handle:
            self.gate_config = json.load(handle)
        self.mean_gate = self.gate_config.get(
            "mean_gate", [1.0 / n_experts] * n_experts
        )
        self.step_separator = step_separator

    # --- gate helpers ----------------------------------------------------- #
    @torch.no_grad()
    def _base_forward(self, input_ids: torch.Tensor, past=None):
        """Forward base thuần (tắt adapter) — đúng input mà gate được train."""
        with self.model.disable_adapter():
            return self.model(
                input_ids=input_ids, past_key_values=past,
                use_cache=True, output_hidden_states=True,
            )

    @torch.no_grad()
    def gate_from_hidden(self, hidden: torch.Tensor) -> List[float]:
        g = self.gate(hidden.reshape(1, -1).float().to(self.device))
        return g[0].tolist()

    # --- generation ------------------------------------------------------- #
    @torch.no_grad()
    def generate(self, question: str, config: SamplingConfig, mode: str = "dynamic") -> str:
        if mode not in {"merged", "per_example", "dynamic"}:
            raise ValueError(f"mode không hợp lệ: {mode}")

        prompt = render_prompt(self.tokenizer, question)
        input_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        input_ids = input_ids["input_ids"].to(self.device)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed)

        if mode == "merged":
            self.mixer.set_mixture(self.mean_gate)

        base_out = self._base_forward(input_ids)
        base_past = base_out.past_key_values
        last_hidden = base_out.hidden_states[-1][0, -1]
        if mode in {"per_example", "dynamic"}:
            self.mixer.set_mixture(self.gate_from_hidden(last_hidden))
        del base_out

        mix_out = self.model(input_ids=input_ids, use_cache=True)
        mix_past = mix_out.past_key_values
        logits = mix_out.logits[0, -1]
        del mix_out

        eos_id = self.tokenizer.eos_token_id
        generated: List[int] = []
        tail = ""

        for _ in range(config.max_new_tokens):
            next_id = _sample(logits, config.temperature, config.top_p, generator)
            if eos_id is not None and next_id == eos_id:
                break
            generated.append(next_id)
            piece = self.tokenizer.decode([next_id], skip_special_tokens=True)
            tail = (tail + piece)[-16:]

            token = torch.tensor([[next_id]], device=self.device)
            base_out = self._base_forward(token, base_past)
            base_past = base_out.past_key_values
            last_hidden = base_out.hidden_states[-1][0, -1]
            del base_out

            if mode == "dynamic" and self.step_separator in tail:
                self.mixer.set_mixture(self.gate_from_hidden(last_hidden))
                tail = ""

            mix_out = self.model(input_ids=token, past_key_values=mix_past, use_cache=True)
            mix_past = mix_out.past_key_values
            logits = mix_out.logits[0, -1]
            del mix_out

        self.mixer.reset()
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def _sample(logits: torch.Tensor, temperature: float, top_p: float, generator) -> int:
    logits = logits.float()
    if temperature <= 0:
        return int(logits.argmax())
    probs = torch.softmax(logits / temperature, dim=-1)
    if 0 < top_p < 1:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        keep = cumulative - sorted_probs < top_p
        keep[0] = True
        sorted_probs = sorted_probs * keep
        sorted_probs = sorted_probs / sorted_probs.sum()
        choice = torch.multinomial(sorted_probs.cpu(), 1, generator=generator)
        return int(sorted_idx[choice].item())
    return int(torch.multinomial(probs.cpu(), 1, generator=generator).item())


# --------------------------------------------------------------------------- #
# Xuất adapter tĩnh cho vLLM (mode = merged)
# --------------------------------------------------------------------------- #
def export_merged_adapter(
    model_name: str,
    phase1_dir: str,
    gate_dir: str,
    out_dir: str,
    n_experts: int = 3,
    weights: Optional[Sequence[float]] = None,
    combination_type: str = "cat",
    verify: bool = True,
    verify_tol: float = 1e-3,
    merge_dtype: str = "float32",
) -> str:
    """Gộp M expert thành **một** adapter theo ḡ để vLLM nạp được.

    PHẢI dùng ``combination_type="cat"``. Ta cần đúng

        dW_mix = sum_m g_m * s_m * B_m A_m

    tức là thứ mà ``AdapterMixer`` tạo ra lúc runtime. ``"cat"`` ghép khối theo
    chiều rank — ``B_cat = [B_1 ... B_M]``, ``A_cat = [w_1 A_1; ...; w_M A_M]``
    — nên ``B_cat A_cat`` **bằng đúng** tổng trên, chỉ đổi rank r -> M*r.

    Ngược lại ``"linear"`` cộng riêng các ``A`` và các ``B`` rồi mới nhân, sinh
    ra các số hạng chéo ``B_m A_n`` (m != n) không có trong tổng cần tính. Nó là
    xấp xỉ, và sai lệch **không gây crash** — chỉ ra số sai. Đó là lý do có
    ``verify``: so delta-weight ``s·B_merged A_merged`` với ``sum_m g_m·s_m·B_m A_m``
    theo từng module ở fp32, lệch quá ``verify_tol`` thì báo lỗi ngay tại chỗ.

    ``merge_dtype`` mặc định ``"float32"``: ``add_weighted_adapter`` tính các khối
    ``cat`` theo dtype của weight, nên gộp trong bf16 nướng sẵn sai số ~2e-3 vào
    file (verify sẽ fail với ``tol=1e-3``). Hạ xuống ``"bfloat16"`` nếu thiếu RAM
    (~28GB cho bản fp32) — khi đó ``verify_tol`` tự nới theo bf16 eps.

    HỆ QUẢ CHO vLLM: rank sau khi gộp là ``M * lora_r`` (mặc định 3*16 = 48), nên
    ``--max_lora_rank`` khi eval phải >= 48. ``eval/run_eval.py`` tự nâng.
    """
    merged_path = os.path.join(out_dir, "merged")
    if os.path.exists(os.path.join(merged_path, "adapter_model.safetensors")):
        print(f"[merge] đã có sẵn -> dùng lại {merged_path} (bỏ qua merge)")
        return merged_path

    tokenizer = load_tokenizer(model_name)
    names = expert_names(n_experts)
    # Gộp ở fp32: add_weighted_adapter tính các khối cat (w_m · A_m) theo dtype
    # của weight, nên gộp trong bf16 sẽ nướng sẵn sai số ~2e-3 vào file adapter —
    # .float() lúc verify không gỡ lại được. File merged to gấp đôi (~9MB), vLLM
    # tự cast bf16 khi nạp.
    dtype = torch.float32 if merge_dtype == "float32" else torch.bfloat16
    base = load_base_model(model_name, dtype=dtype)
    model, adapter_names = load_experts(
        base, [os.path.join(phase1_dir, n) for n in names], names
    )
    if weights is None:
        with open(os.path.join(gate_dir, "gate_config.json"), "r", encoding="utf-8") as handle:
            weights = json.load(handle).get("mean_gate", [1.0 / n_experts] * n_experts)
    weights = [float(w) for w in weights]

    # Tạo adapter gộp TRƯỚC khi động vào scaling của các expert: add_weighted_adapter
    # đọc chính scaling hiện hành, nếu AdapterMixer đã sửa thì sẽ nhân hai lần.
    model.add_weighted_adapter(
        adapters=list(adapter_names),
        weights=weights,
        adapter_name="merged",
        combination_type=combination_type,
    )

    if verify and dtype == torch.bfloat16:
        # bf16 eps ~ 7.8e-3; gộp trong bf16 thì sai số nền cỡ eps/3. Nới tol theo
        # eps thay vì một con số tuỳ tiện. Vẫn bắt được lỗi công thức: 'linear'
        # lệch 0.80, tức gấp 100 lần ngưỡng này.
        verify_tol = max(verify_tol, float(torch.finfo(torch.bfloat16).eps))
        print(f"[merge] gộp ở bf16 -> nới verify_tol = {verify_tol:.2e}")

    max_diff = None
    if verify:
        max_diff = _verify_merge(model, adapter_names, weights, verify_tol)

    model.set_adapter("merged")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, selected_adapters=["merged"])
    tokenizer.save_pretrained(out_dir)
    meta = {
        "weights": weights,
        "combination_type": combination_type,
        "source": os.path.abspath(phase1_dir),
        "verify_max_rel_frobenius": max_diff,
        "merged_rank_multiplier": n_experts if combination_type == "cat" else 1,
    }
    with open(os.path.join(out_dir, "merge_info.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    merged_path = os.path.join(out_dir, "merged")
    print(f"[merge] ḡ = {weights} ({combination_type}) -> {merged_path}")

    del model, base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return merged_path


@torch.no_grad()
def _verify_merge(model, adapter_names: Sequence[str], weights: Sequence[float],
                  tol: float) -> float:
    """So delta-weight của adapter đã gộp với tổ hợp cần tái tạo, ở fp32.

    Bản trước so logits của hai lần forward. Sai lầm: forward chạy bf16, mà bf16
    chỉ có 8 bit mantissa (eps ~ 4e-3), nên hai đường tính khác nhau luôn lệch
    quá 1e-3 dù công thức đúng -> false positive.

    So ở mức weight thì không dính nhiễu dtype, không cần forward, và chính là
    đại lượng ta thực sự quan tâm:

        dW_merged = s_merged * B_merged A_merged
        dW_mix    = sum_m (g_m * s_m) * B_m A_m

    Cố ý materialize dW rồi trừ trực tiếp, KHÔNG dùng đẳng thức r x r như
    core/rep_loss.py: ở đây hai vế gần bằng nhau nên
    ||X-Y||^2 = <X,X> - 2<X,Y> + <Y,Y> sẽ bị catastrophic cancellation, đúng cái
    ta cần đo lại là cái bị mất chính xác nhất.
    """
    worst, worst_name, n_checked = 0.0, "", 0
    for module_name, module in model.named_modules():
        lora_A = getattr(module, "lora_A", None)
        lora_B = getattr(module, "lora_B", None)
        if lora_A is None or lora_B is None:
            continue
        if "merged" not in lora_A or not all(a in lora_A for a in adapter_names):
            continue
        scaling = getattr(module, "scaling", {})
        merged = float(scaling.get("merged", 1.0)) * (
            lora_B["merged"].weight.float() @ lora_A["merged"].weight.float()
        )
        mix = torch.zeros_like(merged)
        for adapter, g in zip(adapter_names, weights):
            coef = float(g) * float(scaling.get(adapter, 1.0))
            mix += coef * (
                lora_B[adapter].weight.float() @ lora_A[adapter].weight.float()
            )
        rel = float((merged - mix).norm() / mix.norm().clamp_min(1e-8))
        n_checked += 1
        if rel > worst:
            worst, worst_name = rel, module_name
        del merged, mix
    if n_checked == 0:
        raise RuntimeError(
            "Không tìm thấy LoRA module nào mang cả 'merged' lẫn M expert — "
            "add_weighted_adapter có thể đã không tạo adapter như mong đợi"
        )
    if worst > tol:
        raise RuntimeError(
            f"Merge KHÔNG tái tạo được sum_m g_m dW_m: lệch Frobenius tương đối "
            f"{worst:.4g} > {tol} (module tệ nhất: {worst_name}, đã kiểm "
            f"{n_checked} module). combination_type hiện tại sai — phải dùng "
            f"'cat'. Nếu đã là 'cat' thì dừng lại và báo, đừng nới tol."
        )
    # AdapterMixer không còn được cover bởi phép so trên -> smoke check riêng,
    # vẫn không cần forward.
    mixer = AdapterMixer(model, adapter_names)
    mixer.set_mixture(weights)
    probe, base_scale = mixer.layers[0]
    for adapter, g in zip(adapter_names, weights):
        expected = base_scale[adapter] * float(g)
        got = float(probe.scaling[adapter])
        if abs(got - expected) > 1e-6:
            mixer.reset()
            raise RuntimeError(
                f"AdapterMixer đặt sai scaling cho {adapter}: {got} != {expected}"
            )
    mixer.reset()
    print(f"[merge] verify OK: lệch Frobenius tương đối tối đa = {worst:.2e} "
          f"trên {n_checked} module; AdapterMixer khớp")
    return worst
