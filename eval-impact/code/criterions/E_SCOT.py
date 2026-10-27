"""E-SCOT: Entropy-aware Span-Constrained Optimal Transport for Cross-Tokenizer KD.

Paper: 15849_Entropy_aware_Span_Const.pdf (ICML 2026).

Loss (Eq. 12 + 10 in the paper):
    L      = (1 - kd_rate) * L_CE + kd_rate * L_E-SCOT
    L_E-SCOT = sum_{i in M} w_i * (1/|N(i)|) * sum_{j in N(i)} d_ij
where:
    d_ij  = ||sort(p_S^i) - sort(p_T^j)||_1               (Eq. 3 ULD metric; 1/ℓ dropped — see compute_escot_loss)
    N(i)  = teacher tokens whose CHAR SPAN overlaps student token i's char span
    w_i   = λ + (1-λ) * |M| * q_i / sum_k q_k              (Eq. 8, stop-grad)
    q_i   = (1/|N(i)|) * sum_{j in N(i)} δ_ij              (Eq. 7)
    δ_ij  = sum_{α in {0.5,1,2}} |Ĥ_α(p_S^i) - Ĥ_α(p_T^j)| (Eq. 7)
    Ĥ_α(p) = (1/log|V|) * (1/(1-α)) * log(sum_k p_k^α)     (Eq. 6, normalized Rényi)
Reference hyperparams: τ=2.0, ρ=0.5, λ=0.25.
"""

import torch
import os
from .cross_entropy_loss import CrossEntropyLoss
from impact_utils import compute_text_offsets


class E_SCOT(CrossEntropyLoss):

    def __init__(self, args, padding_id=-100):
        super().__init__(args, padding_id=padding_id)
        self.args = args
        self.kd_rate = args.kd_rate
        self.kd_temperature = args.kd_temperature
        self.lam = getattr(args, "e_scot_lambda", 0.25)
        alphas = getattr(args, "e_scot_alphas", "0.5,1.0,2.0")
        if isinstance(alphas, str):
            alphas = [float(x.strip()) for x in alphas.split(",")]
        self.alphas = alphas
        print(
            f"----------------Using E-SCOT (λ={self.lam}, α={self.alphas}, "
            f"τ={self.kd_temperature}, ρ={self.kd_rate})----------------------------"
        )

    def forward(self, distiller, input_data, output_data, logging_output, batch_denom):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        self.distiller = distiller
        teacher_key = distiller.teacher_model_type

        student_out = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data.get("position_ids", None),
            output_hidden_states=True,
        )
        with torch.no_grad():
            teacher_model.eval()
            teacher_out = teacher_model(
                input_data[f"teacher_{teacher_key}_input_ids"],
                attention_mask=input_data[f"teacher_{teacher_key}_attention_mask"],
                position_ids=input_data.get(f"teacher_{teacher_key}_position_ids", None),
                output_hidden_states=True,
            )

        log = {}
        loss_ce, nll_loss = self.compute_cross_entropy_loss(
            student_out.logits, output_data["label"], log=log
        )
        log["nll_loss"] = nll_loss

        s_offsets_batch, t_offsets_batch = self._get_offsets(input_data, distiller)

        escot_loss = self.compute_escot_loss(
            student_logits=student_out.logits,
            teacher_logits=teacher_out.logits,
            student_labels=output_data["label"],
            teacher_labels=output_data[f"teacher_{teacher_key}_label"],
            student_offsets_batch=s_offsets_batch,
            teacher_offsets_batch=t_offsets_batch,
        )

        loss = (1.0 - self.kd_rate) * loss_ce + self.kd_rate * escot_loss
        log["escot_loss"] = escot_loss.detach() if torch.is_tensor(escot_loss) else escot_loss
        log["loss"] = loss.detach() if torch.is_tensor(loss) else loss
        log["accuracy"] = self.compute_token_accuracy(
            student_out.logits, output_data["label"],
        )
        logging_output = self.record_logging_output(logging_output, batch_denom, log)
        return loss / batch_denom, logging_output

    def _get_offsets(self, input_data, distiller):
        teacher_key = distiller.teacher_model_type
        # Prefer the dataset's input_ids-aligned separate offsets (compute_separate_offsets).
        # The joint compute_text_offsets fallback below misaligns when the prompt is truncated
        # to max_prompt_length (joint token position != input_ids position), which corrupts the
        # span neighborhoods N(i).
        s_pre = input_data.get("student_offsets")
        t_pre = input_data.get(f"teacher_{teacher_key}_offsets")
        if s_pre is not None and t_pre is not None:
            return s_pre, t_pre

        raw_texts = input_data.get("raw_texts", [])
        teacher_tok = distiller.teacher_tokenizers.get(teacher_key)
        student_tok = distiller.student_tokenizer
        max_len = self.args.max_length

        s_offsets, t_offsets = [], []
        for text in raw_texts:
            if text:
                s_offsets.append(compute_text_offsets(student_tok, text, max_len))
                t_offsets.append(compute_text_offsets(teacher_tok, text, max_len))
            else:
                s_offsets.append([])
                t_offsets.append([])
        return s_offsets, t_offsets

    def compute_escot_loss(
        self,
        student_logits,
        teacher_logits,
        student_labels,
        teacher_labels,
        student_offsets_batch,
        teacher_offsets_batch,
    ):
        """Compute the E-SCOT distillation loss for a batch (SUM-scale)."""
        device = student_logits.device
        B = student_logits.shape[0]
        tau = self.kd_temperature

        # Softmax at temperature τ. Cast to fp32 for stable Rényi/W-1.
        s_probs = torch.softmax(student_logits / tau, dim=-1, dtype=torch.float32)
        t_probs = torch.softmax(teacher_logits / tau, dim=-1, dtype=torch.float32)

        # Sort descending — for 1D Wasserstein-1 on sorted probabilities (ULD metric).
        sorted_s = s_probs.sort(dim=-1, descending=True).values  # [B, L_S, V_S]
        sorted_t = t_probs.sort(dim=-1, descending=True).values  # [B, L_T, V_T]

        V_S = sorted_s.shape[-1]
        V_T = sorted_t.shape[-1]
        L_common = max(V_S, V_T)
        if V_S < L_common:
            sorted_s = torch.cat(
                [sorted_s, sorted_s.new_zeros(B, sorted_s.shape[1], L_common - V_S)], dim=-1
            )
        if V_T < L_common:
            sorted_t = torch.cat(
                [sorted_t, sorted_t.new_zeros(B, sorted_t.shape[1], L_common - V_T)], dim=-1
            )

        # Normalized Rényi entropies per position.
        s_ent = self._compute_renyi_entropies(s_probs, V_S)   # [B, L_S, A]
        t_ent = self._compute_renyi_entropies(t_probs, V_T)   # [B, L_T, A]

        total_loss = student_logits.new_zeros((), dtype=torch.float32)

        _dbg = os.environ.get("ESCOT_DEBUG") and not getattr(self, "_dbg_done", False)

        for b in range(B):
            # Supervised positions = where CE label != padding_id.
            s_valid = student_labels[b] != self.padding_id
            t_valid = teacher_labels[b] != self.padding_id
            if _dbg:
                print(f"[ESCOT_DEBUG] b={b} s_valid={int(s_valid.sum())} "
                      f"t_valid={int(t_valid.sum())} len(s_off)={len(student_offsets_batch[b])} "
                      f"len(t_off)={len(teacher_offsets_batch[b])}", flush=True)
            if s_valid.sum() == 0 or t_valid.sum() == 0:
                if _dbg:
                    print(f"[ESCOT_DEBUG] b={b} CONTINUE: valid==0", flush=True)
                continue

            s_offsets_b = student_offsets_batch[b]
            t_offsets_b = teacher_offsets_batch[b]
            if len(s_offsets_b) == 0 or len(t_offsets_b) == 0:
                if _dbg:
                    print(f"[ESCOT_DEBUG] b={b} CONTINUE: offsets empty", flush=True)
                continue

            s_pos_all = s_valid.nonzero(as_tuple=False).flatten().tolist()
            t_pos_all = t_valid.nonzero(as_tuple=False).flatten().tolist()
            # Only keep positions with a corresponding offset (offsets are prompt+output tokens;
            # the model sees input_ids[:input_len-1] so offsets index the same positions).
            s_pos = [p for p in s_pos_all if p < len(s_offsets_b)]
            t_pos = [p for p in t_pos_all if p < len(t_offsets_b)]
            if _dbg:
                print(f"[ESCOT_DEBUG] b={b} s_pos_all=[{s_pos_all[0] if s_pos_all else '-'}"
                      f"..{s_pos_all[-1] if s_pos_all else '-'}] t_pos_all=[{t_pos_all[0] if t_pos_all else '-'}"
                      f"..{t_pos_all[-1] if t_pos_all else '-'}] len(s_pos)={len(s_pos)} "
                      f"len(t_pos)={len(t_pos)}", flush=True)
            if not s_pos or not t_pos:
                if _dbg:
                    print(f"[ESCOT_DEBUG] b={b} CONTINUE: s_pos/t_pos empty", flush=True)
                continue

            s_spans = [tuple(s_offsets_b[p]) for p in s_pos]
            t_spans = [tuple(t_offsets_b[p]) for p in t_pos]

            neighborhoods = _span_align(s_spans, t_spans)  # list[list[int]]

            pair_i, pair_j = [], []
            for i_local, N_i in enumerate(neighborhoods):
                for j_local in N_i:
                    pair_i.append(i_local)
                    pair_j.append(j_local)
            if _dbg:
                n_nonempty = sum(1 for N_i in neighborhoods if N_i)
                print(f"[ESCOT_DEBUG] b={b} s_spans[0:2]={s_spans[:2]} t_spans[0:2]={t_spans[:2]} "
                      f"neighborhoods_nonempty={n_nonempty}/{len(neighborhoods)} pairs={len(pair_i)}",
                      flush=True)
            if not pair_i:
                if _dbg:
                    print(f"[ESCOT_DEBUG] b={b} CONTINUE: no pairs", flush=True)
                continue

            s_pos_t = torch.tensor(s_pos, device=device, dtype=torch.long)
            t_pos_t = torch.tensor(t_pos, device=device, dtype=torch.long)
            pair_i_t = torch.tensor(pair_i, device=device, dtype=torch.long)
            pair_j_t = torch.tensor(pair_j, device=device, dtype=torch.long)

            s_sort_sel = sorted_s[b].index_select(0, s_pos_t)  # [I, ℓ]
            t_sort_sel = sorted_t[b].index_select(0, t_pos_t)  # [J, ℓ]
            s_ent_sel = s_ent[b].index_select(0, s_pos_t)      # [I, A]
            t_ent_sel = t_ent[b].index_select(0, t_pos_t)      # [J, A]

            # d_ij for each pair, DIFFERENTIABLE through student.
            s_pairs = s_sort_sel.index_select(0, pair_i_t)     # [P, ℓ]
            t_pairs = t_sort_sel.index_select(0, pair_j_t)     # [P, ℓ]
            # W1 ground metric (Eq.3) WITHOUT the 1/ℓ — matches DSKD ULD.py; with 1/ℓ (ℓ=|V|~32k)
            # this sum-scale term is ~1e5x below L_CE and the KD signal vanishes.
            d_ij = (s_pairs - t_pairs).abs().sum(dim=-1)  # [P]

            # δ_ij (stop-grad — used only in weight construction).
            s_ent_p = s_ent_sel.index_select(0, pair_i_t)      # [P, A]
            t_ent_p = t_ent_sel.index_select(0, pair_j_t)      # [P, A]
            delta_ij = (s_ent_p - t_ent_p).abs().sum(dim=-1).detach()  # [P]

            # q_i = (1/|N(i)|) * sum_j δ_ij, over pairs indexed by pair_i_t.
            I = len(neighborhoods)
            counts = torch.zeros(I, device=device, dtype=torch.float32)
            delta_sum = torch.zeros(I, device=device, dtype=torch.float32)
            counts.scatter_add_(0, pair_i_t, torch.ones_like(pair_i_t, dtype=torch.float32))
            delta_sum.scatter_add_(0, pair_i_t, delta_ij)
            valid = counts > 0
            q = torch.zeros_like(counts)
            q[valid] = delta_sum[valid] / counts[valid]

            # w_i = λ + (1-λ) * |M| * q_i / sum_k q_k.
            M_mask = valid
            M_size = M_mask.sum().float().clamp(min=1.0)
            q_sum = q[M_mask].sum().clamp(min=1e-8)
            w = torch.ones_like(q)
            w[M_mask] = self.lam + (1.0 - self.lam) * M_size * q[M_mask] / q_sum
            w = w.detach()  # stop-grad through weights (paper §3.4)

            # L_E-SCOT (per batch item):
            #   sum_i w_i * (1/|N(i)|) * sum_{j in N(i)} d_ij
            #   = sum_over_pairs (w[i] / |N(i)|) * d_ij
            per_pair_w = w.index_select(0, pair_i_t) / counts.index_select(0, pair_i_t).clamp(min=1.0)
            item_loss = (per_pair_w * d_ij).sum()
            total_loss = total_loss + item_loss

        if _dbg:
            print(f"[ESCOT_DEBUG] total_loss={float(total_loss):.6f}", flush=True)
            self._dbg_done = True

        return total_loss

    def _compute_renyi_entropies(self, probs, V):
        """Normalized Rényi entropies for orders in self.alphas."""
        p = probs.clamp(min=1e-12)
        log_V = torch.log(torch.tensor(float(V), device=probs.device)) if V > 1 else torch.tensor(1.0, device=probs.device)
        outputs = []
        for alpha in self.alphas:
            if abs(alpha - 1.0) < 1e-6:
                # Shannon (α=1 limit): H_1(p) = -sum p log p, normalized by log|V|.
                H = -(p * p.log()).sum(dim=-1) / log_V
            else:
                # H_α = (1/(1-α)) * log(sum p^α), normalized by 1/log|V|.
                # Use logsumexp on α * log p for numerical stability.
                lse = torch.logsumexp(alpha * p.log(), dim=-1)
                H = lse / (1.0 - alpha) / log_V
            outputs.append(H)
        return torch.stack(outputs, dim=-1)


def _span_align(s_spans, t_spans):
    """Algorithm 1 of paper 15849 — linear two-pointer sweep for character-span overlap.

    Returns: list of length |s_spans|; entry i is the list of teacher indices whose span
             [t_start, t_end) has non-empty intersection with student span [s_start, s_end).
    """
    neighborhoods = [[] for _ in range(len(s_spans))]
    if not t_spans:
        return neighborhoods
    j0 = 0
    for i, (st_i, ed_i) in enumerate(s_spans):
        # Advance j0 past teacher tokens that fully precede student token i.
        while j0 < len(t_spans) and t_spans[j0][1] <= st_i:
            j0 += 1
        # Collect all teacher tokens whose start is before student i's end.
        k = j0
        while k < len(t_spans) and t_spans[k][0] < ed_i:
            neighborhoods[i].append(k)
            k += 1
    return neighborhoods
