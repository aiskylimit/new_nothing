"""Trainer plumbing for L = L_NLL + lambda * L_trans (mixin over MaskedSFTTrainer).

The base class owns the NLL (vanilla or SGL-masked, Eq. 9); this mixin adds the transition term
on top of *every* kept transition pair regardless of the NLL mask, scales it so lambda means
"per optimizer step" under gradient accumulation, gives the predictor its own optimizer group
and logs the diagnostics the method spec asks for.
"""

import json
import os
from collections import defaultdict

import torch
from transformers import TrainerCallback

from sgl.training.transition_loss import HiddenStateCapture, TransitionPredictor, find_decoder_layers, transition_loss

PREDICTOR_FILE = "trans_predictor.pt"
PREDICTOR_ATTR = "trans_predictor"
TRANS_INPUT_KEYS = ("step_id", "step_end", "pair_src", "num_steps")


def attach_predictor(model, hidden: int) -> TransitionPredictor:
    """Register the predictor as a submodule of the (PEFT) model.

    A submodule, not a free-standing module: DeepSpeed/Trainer only move, checkpoint and
    optimize parameters that live under the model they were handed. PEFT's adapter save ignores
    it (not an adapter weight), so adapter checkpoints stay loadable by vLLM as before; the
    predictor is written next to them as trans_predictor.pt by PredictorCheckpointCallback.
    """
    d_model = model.config.hidden_size
    predictor = TransitionPredictor(d_model, hidden).float()
    predictor.to(next(model.parameters()).device)
    setattr(model, PREDICTOR_ATTR, predictor)
    return predictor


def predictor_of(model) -> TransitionPredictor:
    return getattr(model, PREDICTOR_ATTR)


class PredictorCheckpointCallback(TrainerCallback):
    """Save the predictor into every checkpoint-N/ (rank 0), so --resume can pick it up."""

    def __init__(self, predictor: torch.nn.Module):
        self.predictor = predictor

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}", PREDICTOR_FILE)
            torch.save(self.predictor.state_dict(), path)


class _HiddenGradProbe:
    """||dL_NLL/dH|| vs lambda*||dL_trans/dH|| at the transition layer, from the real backward.

    A separate backward per term is not an option under DeepSpeed (its output hooks treat any
    autograd pass through the logits as *the* backward and run the ZeRO reduce epilogue), so the
    split is read off tensor hooks instead: the source states see only the transition term, the
    layer output H sees both, and the NLL share is the difference. The ratio at H stands in for
    the LoRA-parameter ratio of every layer <= the transition layer (their gradients are linear in
    dL/dH); with max_grad_norm=1 and batch 1, a ratio persistently above ~0.3-0.5 means the
    auxiliary term is crowding out the NLL update: lower lambda.
    """

    def __init__(self, add_metric):
        self._add_metric = add_metric
        self._trans = []  # (example index, positions [P], grad [P, d]) per example

    def source_hook(self, example: int):
        def hook(positions, grad):
            self._trans.append((example, positions, grad))

        return hook

    def hidden_hook(self, grad):
        # ||g - s||^2 = ||g||^2 - 2<g_pos, s> + ||s||^2 with s the transition share scattered at
        # the source positions: no full-size copy of the [B, T, d] gradient is needed.
        total_sq = grad.float().pow(2).sum()
        trans_sq = grad.new_zeros((), dtype=torch.float32)
        cross = grad.new_zeros((), dtype=torch.float32)
        for example, positions, share in self._trans:
            share = share.float()
            trans_sq += share.pow(2).sum()
            cross += (grad[example, positions].float() * share).sum()
        nll_sq = (total_sq - 2 * cross + trans_sq).clamp_min(0.0)
        g_nll, g_trans = float(nll_sq.sqrt()), float(trans_sq.sqrt())
        self._add_metric("grad_nll_hidden", g_nll)
        self._add_metric("grad_trans_hidden", g_trans)
        self._add_metric("grad_trans_ratio", g_trans / max(g_nll, 1e-12))
        self._trans.clear()


class TransitionLossMixin:
    """Mix in *before* MaskedSFTTrainer: `class TransitionSFTTrainer(TransitionLossMixin, MaskedSFTTrainer)`.

    Constructor kwargs (all popped before reaching Trainer):
        trans_lambda: weight of L_trans at full strength.
        trans_layer: 0-based decoder layer whose output is used for source and target.
        trans_lr: learning rate of the predictor's param group (no weight decay).
        trans_lambda_warmup: ramp lambda linearly from 0 over the LR warmup steps.
        trans_shuffle_targets: the shuffled-target control.
        trans_grad_log_interval: on every Nth optimizer step, measure ||dL_NLL/dH|| and
            lambda*||dL_trans/dH|| at the transition layer's hidden state H (tensor hooks in the
            real backward, no extra passes; 0 disables).
    """

    def __init__(self, *args, **kwargs):
        self.trans_lambda = float(kwargs.pop("trans_lambda"))
        self.trans_layer = int(kwargs.pop("trans_layer"))
        self.trans_lr = float(kwargs.pop("trans_lr"))
        self.trans_lambda_warmup = bool(kwargs.pop("trans_lambda_warmup", True))
        self.trans_shuffle_targets = bool(kwargs.pop("trans_shuffle_targets", False))
        self.trans_grad_log_interval = int(kwargs.pop("trans_grad_log_interval", 50))
        super().__init__(*args, **kwargs)
        # Registered on the raw layer object before DeepSpeed/DDP wrap the model; wrappers
        # keep the same module instances, so the hook keeps firing.
        layers = find_decoder_layers(self.model)
        if not 0 <= self.trans_layer < len(layers):
            raise ValueError(f"--trans-layer {self.trans_layer} out of range for {len(layers)} layers")
        self._capture = HiddenStateCapture(layers[self.trans_layer])
        self._predictor = predictor_of(self.model)
        self._metric_sums = defaultdict(float)
        self._metric_counts = defaultdict(int)
        self.add_callback(PredictorCheckpointCallback(self._predictor))

    # ---- optimizer: predictor in its own group (own LR, no weight decay) ----
    def create_optimizer(self, model=None):
        opt_model = self.model if model is None else model
        if self.optimizer is None:
            decay_names = set(self.get_decay_parameter_names(opt_model))
            groups = {"decay": [], "no_decay": [], "predictor": []}
            for name, param in opt_model.named_parameters():
                if not param.requires_grad:
                    continue
                if PREDICTOR_ATTR in name:
                    groups["predictor"].append(param)
                elif name in decay_names:
                    groups["decay"].append(param)
                else:
                    groups["no_decay"].append(param)
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
            param_groups = [
                {"params": groups["decay"], "weight_decay": self.args.weight_decay},
                {"params": groups["no_decay"], "weight_decay": 0.0},
                {"params": groups["predictor"], "weight_decay": 0.0, "lr": self.trans_lr},
            ]
            self.optimizer = optimizer_cls([g for g in param_groups if g["params"]], **optimizer_kwargs)
        return self.optimizer

    # ---- lambda schedule ----
    def current_lambda(self) -> float:
        if not self.trans_lambda_warmup:
            return self.trans_lambda
        warmup = self.args.get_warmup_steps(self.state.max_steps) if self.state.max_steps else 0
        if warmup <= 0:
            return self.trans_lambda
        return self.trans_lambda * min(1.0, self.state.global_step / warmup)

    # ---- loss ----
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        trans_inputs = {key: inputs.pop(key) for key in TRANS_INPUT_KEYS if key in inputs}
        self._capture.arm()
        loss_nll, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch, **kwargs
        )
        hidden = self._capture.take()  # [B, T, d]

        probe = self._grad_probe() if self.model.training else None
        losses, stats_list = [], []
        for b in range(hidden.size(0)):
            step_end = trans_inputs["step_end"][b]
            pair_src = trans_inputs["pair_src"][b]
            loss_b, stats_b = transition_loss(
                hidden[b],
                trans_inputs["step_id"][b],
                step_end[step_end >= 0],
                pair_src[pair_src >= 0],
                self._predictor,
                shuffle_targets=self.trans_shuffle_targets,
                source_grad_hook=probe.source_hook(b) if probe else None,
            )
            losses.append(loss_b)
            stats_list.append(stats_b)
        loss_trans = torch.stack(losses).mean()
        if probe:
            hidden.register_hook(probe.hidden_hook)

        # The NLL is already sum/Z over the whole optimizer step (model_accepts_loss_kwargs), so
        # Trainer does not divide by the accumulation count; L_trans is a per-microbatch mean and
        # must be, or lambda would silently scale with --gradient-accumulation-steps.
        lam = self.current_lambda()
        accumulation = getattr(self, "current_gradient_accumulation_steps", self.args.gradient_accumulation_steps)
        loss_trans_scaled = lam * loss_trans / accumulation
        loss = loss_nll + loss_trans_scaled

        if self.model.training:
            # loss_nll here is one microbatch's share of the step-level sum/Z; scaling by the
            # accumulation count puts it on the same per-step scale as loss_trans and `loss`.
            self._record(loss_nll * accumulation, loss_trans, stats_list, lam)
        return (loss, outputs) if return_outputs else loss

    # ---- diagnostics ----
    def _record(self, loss_nll, loss_trans, stats_list, lam):
        scored = [s for s in stats_list if s.pairs > 0]
        values = {
            "loss_nll": float(loss_nll.detach()),
            "loss_trans": float(loss_trans.detach()),
            "trans_lambda": lam,
            "trans_pairs": float(sum(s.pairs for s in stats_list)) / max(1, len(stats_list)),
        }
        if scored:
            for key in ("cos", "copy_cos", "raw_step_cos", "ztilde_norm"):
                values[f"trans_{key}"] = sum(getattr(s, key) for s in scored) / len(scored)
        for key, value in values.items():
            self._metric_sums[key] += value
            self._metric_counts[key] += 1

    def _grad_probe(self):
        interval = self.trans_grad_log_interval
        if interval <= 0 or self.state.global_step % interval != 0:
            return None
        return _HiddenGradProbe(self._add_metric)

    def _add_metric(self, key: str, value: float) -> None:
        self._metric_sums[key] += value
        self._metric_counts[key] += 1

    def log(self, logs, *args, **kwargs):
        # Rank-local means over the microbatches since the previous log line (DeepSpeed prints
        # rank 0 only); merged into the same dict as loss/grad_norm/lr so they land in log_history.
        for key, total in self._metric_sums.items():
            logs[key] = round(total / max(1, self._metric_counts[key]), 6)
        self._metric_sums.clear()
        self._metric_counts.clear()
        super().log(logs, *args, **kwargs)

    # ---- persistence ----
    def save_predictor(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self._predictor.state_dict(), os.path.join(output_dir, PREDICTOR_FILE))
        (json.dump(
            {
                "trans_lambda": self.trans_lambda,
                "trans_layer": self.trans_layer,
                "trans_lr": self.trans_lr,
                "trans_lambda_warmup": self.trans_lambda_warmup,
                "trans_shuffle_targets": self.trans_shuffle_targets,
                "predictor_hidden": self._predictor.up.out_features,
                "d_model": self._predictor.up.in_features,
            },
            open(os.path.join(output_dir, "trans-config.json"), "w"),
            indent=2,
        ))

    def load_predictor(self, checkpoint_dir: str) -> bool:
        path = os.path.join(checkpoint_dir, PREDICTOR_FILE)
        if not os.path.isfile(path):
            return False
        state = torch.load(path, map_location=next(self._predictor.parameters()).device)
        self._predictor.load_state_dict(state)
        return True
