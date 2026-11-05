"""Task weighting for the multi-task students (--mto). The batch loss is split by task (the rows' "task"
field: ee, ner, re). Each task's loss is computed on that task's rows alone, so it is a token mean over
those rows only, and the task losses are combined:
  equal: plain mean over the tasks present in the micro-batch
  famo:  FAMO (Liu et al., NeurIPS 2023), after the authors' code (github.com/Cranial-XIX/FAMO,
         methods/weight_methods.py): one logit per task, loss = sum_k z_k log L_k / c with z = softmax and
         c = sum_k z_k / L_k, so task k's gradient is scaled by z_k / (c L_k). After each optimizer step the
         same batch is scored again and the logits move toward the tasks whose log loss fell least.
--mto none (default) leaves the old loss alone: one token mean over the whole batch, so a task's weight is
its share of the batch's answer tokens."""
import torch
import torch.distributed as dist
import torch.nn.functional as F


def task_rows(tasks, device):
    """{task: LongTensor of the micro-batch rows of that task}, tasks in sorted order"""
    rows = {}
    for i, t in enumerate(tasks):
        rows.setdefault(t, []).append(i)
    return {t: torch.tensor(r, device=device) for t, r in sorted(rows.items())}


class TaskWeighting:
    def __init__(self, args, tasks, device):
        assert args.mto in ("equal", "famo"), args.mto
        assert len(tasks) > 1, f"--mto {args.mto} needs rows of several tasks, the train rows have {tasks}"
        self.mode = args.mto
        self.tasks = sorted(tasks)
        self.prev = {}     # this optimizer step's task losses before the update, one per micro-batch
        self.cache = []    # this step's micro-batches, scored again after the update (FAMO)
        self.logged = {}   # task losses since the last log line
        if self.mode == "famo":
            # the logits would drift apart across processes: every run here uses one GPU
            assert dist.get_world_size() == 1, "--mto famo runs on one GPU"
            self.logits = torch.zeros(len(self.tasks), device=device, requires_grad=True)
            self.opt = torch.optim.Adam([self.logits], lr=args.famo_lr, weight_decay=args.famo_gamma)

    def weights(self, present):
        """{task: weight} over the tasks present in a micro-batch, summing to 1"""
        if self.mode == "equal":
            return {t: 1 / len(present) for t in present}
        idx = torch.tensor([self.tasks.index(t) for t in present], device=self.logits.device)
        z = F.softmax(self.logits.detach()[idx], -1)
        return dict(zip(present, z))

    def combine(self, losses):
        """losses = {task: loss of that task's rows}; returns the micro-batch loss to backpropagate"""
        for t, l in losses.items():
            self.prev.setdefault(t, []).append(l.detach())
            self.logged.setdefault(t, []).append(l.detach())
        z = self.weights(sorted(losses))
        if self.mode == "equal":
            return sum(z[t] * losses[t] for t in losses)
        D = {t: losses[t] + 1e-8 for t in losses}
        c = sum(z[t] / D[t].detach() for t in losses)
        return sum(z[t] * D[t].log() for t in losses) / c

    def update(self, curr):
        """FAMO, after the optimizer step: curr = {task: [loss of each of this step's micro-batches, scored
        again with the updated model]}. Tasks missing from the step are left out of the softmax."""
        present = [t for t in self.tasks if t in self.prev and t in curr]
        delta = torch.stack([(torch.stack(self.prev[t]).mean() + 1e-8).log() - (torch.stack(curr[t]).mean() + 1e-8).log()
                             for t in present])
        idx = torch.tensor([self.tasks.index(t) for t in present], device=self.logits.device)
        with torch.enable_grad():
            d = torch.autograd.grad(F.softmax(self.logits[idx], -1), self.logits, grad_outputs=delta.detach())[0]
        self.opt.zero_grad()
        self.logits.grad = d
        self.opt.step()
        self.prev = {}

    def end_step(self):
        """equal: nothing to learn, only drop the step's losses"""
        self.prev = {}

    def log_line(self):
        """mean task losses since the last call and the current weights over all tasks"""
        z = self.weights(self.tasks)
        parts = []
        for t in self.tasks:
            loss = torch.stack(self.logged[t]).mean().item() if t in self.logged else float("nan")
            parts.append(f"{t} loss {loss:.4f} w {float(z[t]):.3f}")
        self.logged = {}
        return f"mto {self.mode} | " + " | ".join(parts)
