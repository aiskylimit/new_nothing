"""E-SCOT + IMPACT: E-SCOT distillation loss + IMPACT's BPIA hidden-state alignment.

Loss = (1 - kd_rate) * L_CE + kd_rate * L_E-SCOT + lambda_impact * L_BPIA
"""

import torch
from .E_SCOT import E_SCOT
from impact_utils import IMPACTModule


class E_SCOT_IMPACT(E_SCOT):

    def __init__(self, args, padding_id=-100):
        super().__init__(args, padding_id=padding_id)
        self.lambda_impact = getattr(args, "impact_lambda", 1.0)
        self.impact = IMPACTModule(args)

    def forward(self, distiller, input_data, output_data, logging_output, batch_denom):
        self.impact.prepare(distiller)
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

        impact_loss = self.impact.compute_loss(
            distiller,
            list(student_out.hidden_states),
            list(teacher_out.hidden_states),
            s_offsets_batch, t_offsets_batch,
            input_data["attention_mask"],
            input_data[f"teacher_{teacher_key}_attention_mask"],
        )

        loss = (
            (1.0 - self.kd_rate) * loss_ce
            + self.kd_rate * escot_loss
            + self.lambda_impact * impact_loss
        )
        log["escot_loss"] = escot_loss.detach() if torch.is_tensor(escot_loss) else escot_loss
        log["impact_loss"] = impact_loss.detach() if torch.is_tensor(impact_loss) else impact_loss
        log["loss"] = loss.detach() if torch.is_tensor(loss) else loss
        log["accuracy"] = self.compute_token_accuracy(
            student_out.logits, output_data["label"],
        )
        logging_output = self.record_logging_output(logging_output, batch_denom, log)
        return loss / batch_denom, logging_output
