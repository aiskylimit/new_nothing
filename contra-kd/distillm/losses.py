import torch
import torch.nn.functional as F


def entropy_from_logits(logits):
    log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def masked_mean(values, mask):
    mask = mask.to(values.device).float()
    denom = mask.sum().clamp_min(1e-8)
    return (values * mask).sum() / denom


def normalize_entropy(entropy):
    return entropy / entropy.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def top_k_entropy_position(norm_entropy, top_p=0.1):
    cum_entropy = torch.cumsum(norm_entropy, dim=-1)
    return (cum_entropy >= top_p).float().argmax(dim=-1)


def forward_kl(logits, teacher_logits, no_model_batch):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def reverse_kl(logits, teacher_logits, no_model_batch):
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def symmetric_kl(logits, teacher_logits, no_model_batch, lam=0.9):
    for_kl = forward_kl(logits, teacher_logits, no_model_batch)
    rev_kl = reverse_kl(logits, teacher_logits, no_model_batch)
    distil_loss = (1-lam) * for_kl + lam * rev_kl
    return distil_loss
    
def js_distance(logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1-lam) * teacher_probs + lam * student_probs

    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = lam * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss += (1-lam) * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss
    
def tv_distance(logits, teacher_logits, no_model_batch):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    
    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    prod_probs = 0.5 * torch.masked_fill(torch.abs(teacher_probs - student_probs), inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = lam * teacher_probs + (1-lam) * student_probs
    mixed_logprobs = torch.log(mixed_probs)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    
    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1-lam) * teacher_probs + lam * student_probs
    
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def indirect_kd_loss(student_logits, teacher_logits, mask=None, temperature=1.0, top_k=10):
    """Listwise ranking loss over the student's top-k token candidates."""
    vocab_size = student_logits.size(-1)
    top_k = max(1, min(top_k, vocab_size))

    student_logits = student_logits.float() / temperature
    teacher_logits = teacher_logits.float() / temperature

    student_topk_logits, student_topk_indices = torch.topk(student_logits, top_k, dim=-1)
    teacher_topk_logits = torch.gather(teacher_logits, dim=-1, index=student_topk_indices)
    teacher_sorted_idx = torch.argsort(teacher_topk_logits, dim=-1, descending=True)
    sorted_student_logits = torch.gather(student_topk_logits, dim=-1, index=teacher_sorted_idx)
    sorted_student_logits = sorted_student_logits - sorted_student_logits.mean(dim=-1, keepdim=True)

    log_cumsum_exp = torch.logcumsumexp(sorted_student_logits, dim=-1)
    token_loss = -(sorted_student_logits - log_cumsum_exp).sum(dim=-1)

    if mask is None:
        return token_loss.mean()
    return masked_mean(token_loss, mask)


def tsd_kd_loss(
    logits,
    teacher_logits,
    no_model_batch,
    beta=0.9,
    temperature=1.0,
    token_entropy_percentile_threshold=0.1,
    indirect_kd_alpha=0.1,
    entropy_alpha=1.0,
    top_k=10,
):
    """
    Token-Selective Dual KD objective ported from the TSD-KD reference trainer.

    The direct term is generalized JSD weighted by relative student/teacher
    entropy. The auxiliary terms are high-entropy regularization and the
    teacher-ranked top-k indirect KD loss.
    """
    labels = no_model_batch["label"]
    mask = labels != -100

    student_logits = logits.float() / temperature
    teacher_logits = teacher_logits.float() / temperature

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    beta = torch.as_tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
    beta = beta.clamp(1e-6, 1.0 - 1e-6)
    mixture_log_probs = torch.logsumexp(
        torch.stack(
            [
                student_log_probs + torch.log1p(-beta),
                teacher_log_probs + torch.log(beta),
            ],
            dim=0,
        ),
        dim=0,
    )

    kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True).sum(dim=-1)
    kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True).sum(dim=-1)

    student_entropy = entropy_from_logits(student_logits)
    teacher_entropy = entropy_from_logits(teacher_logits)
    confidence_weight = torch.sigmoid(student_entropy - teacher_entropy).detach()

    jsd = beta * kl_teacher + (1.0 - beta) * kl_student
    direct_loss = masked_mean(jsd * confidence_weight, mask)

    valid_entropies = student_entropy[mask]
    if valid_entropies.numel() == 0:
        entropy_reg = student_entropy.sum() * 0.0
    else:
        token_entropy_percentile_threshold = min(max(token_entropy_percentile_threshold, 0.0), 1.0)
        threshold = torch.quantile(valid_entropies.float(), token_entropy_percentile_threshold)
        entropy_mask = mask & (student_entropy >= threshold)
        entropy_reg = masked_mean(student_entropy, entropy_mask)

    normalized_entropies = normalize_entropy(valid_entropies.float()) if valid_entropies.numel() > 0 else None
    if normalized_entropies is None:
        topk_loss = student_logits.sum() * 0.0
    else:
        indirect_token_len = int(top_k_entropy_position(
            normalized_entropies,
            token_entropy_percentile_threshold,
        ).item()) + 1
        indirect_token_len = min(indirect_token_len, student_logits.size(1))
        topk_loss = indirect_kd_loss(
            student_logits[:, :indirect_token_len, :],
            teacher_logits[:, :indirect_token_len, :],
            mask=mask[:, :indirect_token_len],
            temperature=1.0,
            top_k=top_k,
        )

    return direct_loss + entropy_alpha * entropy_reg + indirect_kd_alpha * topk_loss


def l2_loss_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Computes normalized L2 loss only for valid (non-padded) tokens."""
    # Compute mean squared error per token (averaged over vocab dimension)
    mse_per_token = F.mse_loss(pred, target, reduction='none').mean(dim=-1)  # [B, L]
    # Apply mask and average over valid tokens
    masked_losses = mse_per_token * mask.float()
    valid_tokens = mask.sum()
    if valid_tokens > 0:
        return masked_losses.sum() / valid_tokens
    else:
        return torch.tensor(0.0, device=pred.device)


def cosine_similarity_loss_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Computes masked cosine similarity loss."""
    # Compute cosine similarity per token
    cos_sim = F.cosine_similarity(pred.float(), target.float(), dim=-1)  # [B, L]
    # Apply mask and compute loss for valid tokens only
    masked_cos_sim = cos_sim * mask.float()
    valid_tokens = mask.sum()
    if valid_tokens > 0:
        return (1 - masked_cos_sim.sum() / valid_tokens)
    else:
        return torch.tensor(0.0, device=pred.device)


def hybrid_loss_masked(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    mask: torch.Tensor, 
    cosine_weight: float = 0.6, 
    l2_weight: float = 0.4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Hybrid loss combining cosine similarity and L2 loss.
    
    Args:
        pred: Predicted logits [B, L, V]
        target: Target logits [B, L, V] 
        mask: Attention mask [B, L]
        cosine_weight: Weight for cosine similarity loss (emphasizes direction)
        l2_weight: Weight for L2 loss (emphasizes magnitude)
    
    Returns:
        Combined loss value, cosine loss, l2 loss
    """
    # Cosine similarity loss (for directional alignment)
    cosine_loss = cosine_similarity_loss_masked(pred, target, mask)
    
    # L2 loss (for magnitude preservation)
    l2_loss = l2_loss_masked(pred, target, mask)
    
    # Combine with weights
    hybrid_loss = cosine_weight * cosine_loss + l2_weight * l2_loss
    
    return hybrid_loss, cosine_loss, l2_loss


def cosine_similarity_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes 1 - mean(cosine_similarity(a, b))."""
    return (1 - F.cosine_similarity(a.float(), b.float(), dim=-1)).mean()


def velocity_field_loss(
    student_hiddens, 
    teacher_hiddens,
    velocity_field, 
    projector,
    teacher_schedule,
    student_schedule,
    attention_mask,
    device=0
):
    """
    Compute FRFD velocity field loss for rectified flow distillation.
    """
    total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    # Train only at the student endpoint, matching the distillation-time query.
    batch_size = student_hiddens[0].size(0)
    t = torch.zeros(batch_size, 1, 1, device=device, dtype=torch.float32)
    num_distill_layers = len(teacher_schedule)
    
    # Loop over all distillation layers
    for j, (teacher_layer_idx, student_layer_idx) in enumerate(zip(teacher_schedule, student_schedule)):
        # Get hidden states for the current layer pair
        y_S = student_hiddens[student_layer_idx].to(device=f"cuda:{device}", dtype=torch.float32)
        y_T = teacher_hiddens[teacher_layer_idx].to(device=f"cuda:{device}", dtype=torch.float32)
        
        # Project student features to teacher's dimension
        y_S = projector(y_S)

        # Create interpolated features Y_t
        Y_t = (1 - t) * y_S + t * y_T
        
        # Compute target velocity
        target_velocity = y_T - y_S
        
        # Predict velocity using the velocity field model
        layer_indices = torch.tensor([j] * y_S.size(0), device=device, dtype=torch.long)
        predicted_velocity = velocity_field(Y_t, t.squeeze(1).squeeze(1), layer_indices)
        
        # Accumulate the MSE loss for this layer
        loss_per_token = F.mse_loss(predicted_velocity, target_velocity, reduction='none').mean(dim=-1)
        loss_per_token *= attention_mask
        loss = loss_per_token.sum() / attention_mask.sum()
        total_loss += loss / num_distill_layers
    
    return total_loss


def frfd_distillation_loss(
    student_hiddens,
    velocity_field,
    projector,
    student_schedule,
    attention_mask,
    num_distill_layers,
    cur_t=1,
    device=0
):
    """
    Compute FRFD rectified flow distillation loss.
    Exactly matches the original FRFD stage2 implementation.
    """
    # Calculate Rectified Flow Distillation Loss
    loss_rfd = 0
    # delta_t = 1.0 / (num_distill_layers - 1)
    
    if attention_mask.sum() > 0:
        for j in range(num_distill_layers):
            h_S_current = student_hiddens[student_schedule[j]].to(device=f"cuda:{device}", dtype=torch.float32)
            # h_S_next = student_hiddens[student_schedule[j+1]].to(device=f"cuda:{device}", dtype=torch.float32)
            
            actual_y_next = projector(h_S_current)
            with torch.no_grad():
                y_S_j = projector(h_S_current.detach())
                B, L, V = y_S_j.shape
                
                # Get ideal update from velocity field
                t = torch.full((B,), 0, device=device, dtype=torch.float32)
                layer_indices = torch.full((B,), j, device=device, dtype=torch.long)
                ideal_update = velocity_field(y_S_j, t, layer_indices)
                
                # Target for next layer: Euler step from current layer
                target_y_next = y_S_j + cur_t * ideal_update #* delta_t
            
            layer_loss = cosine_similarity_loss_masked(actual_y_next, target_y_next, attention_mask)
            loss_rfd += layer_loss / (num_distill_layers)
            
    else:
        loss_rfd = torch.tensor(0.0, device=device)
    
    return loss_rfd

def soft_label_distill_loss(student_logits, teacher_logits, mask, distill_temperature = 2.0):
    student_probs = F.log_softmax(student_logits / distill_temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / distill_temperature, dim=-1)

    loss = F.kl_div(student_probs, teacher_probs, reduction='none').sum(dim=-1)
    loss = (loss * mask).sum() / mask.sum()

    return loss

def get_fdd_loss(t_hiddens, s_hiddens, mask, teacher, student, teacher_schedule, student_schedule):
    # mask = (no_model_batch["label"] != -100).int()
    
    assert len(teacher_schedule) == len(student_schedule), "Mismatch in layer scheduling between teacher and student!!!"
    s_hiddens_selected = torch.stack([s_hiddens[idx] for idx in student_schedule], dim=0)
    t_hiddens_selected = torch.stack([t_hiddens[idx] for idx in teacher_schedule], dim=0)
    
    try:
        # if args.model_type == 'opt':
        #     s_decoder_proj = student.module.model.model.decoder.project_out
        #     if s_decoder_proj is not None:
        #         s_hidden = s_decoder_proj(s_hiddens_selected)

        #     t_decoder_proj = teacher.model.decoder.project_out
        #     if t_decoder_proj is not None:
        #         t_hidden = t_decoder_proj(t_hiddens_selected)
        s_hiddens_logits = student.module.lm_head(s_hiddens_selected)
        t_hiddens_logits = teacher.lm_head(t_hiddens_selected)
        traj_loss = soft_label_distill_loss(s_hiddens_logits, t_hiddens_logits, mask.unsqueeze(0)) / len(teacher_schedule)
        
        s_hiddens_logs = F.log_softmax(s_hiddens_logits, dim=-1)
        t_hiddens_logs = F.log_softmax(t_hiddens_logits, dim=-1)
        delta_hiddens_student = s_hiddens_logs[1:] - s_hiddens_logs[:-1]
        delta_hiddens_teacher = t_hiddens_logs[1:] - t_hiddens_logs[:-1]
        cos_sim = F.cosine_similarity(delta_hiddens_student, delta_hiddens_teacher, dim=-1, eps=1e-5)
        der_loss = 1 - cos_sim
        der_loss = (der_loss * mask.unsqueeze(0)).sum() / (mask.sum() * (len(teacher_schedule)-1))
        
        return traj_loss + der_loss
    
    except torch.OutOfMemoryError:
        try:
            del s_hiddens_logits
            del t_hiddens_logits
            del traj_loss
            del s_hiddens_logs
            del t_hiddens_logs
            del delta_hiddens_student
            del delta_hiddens_teacher
            del cos_sim
            del der_loss
        except:
            pass
        torch.cuda.empty_cache()
        i = 0
        traj_loss, der_loss = 0.0, 0.0
        pre_s_hidden_logs, pre_t_hidden_logs = None, None
        for s_hidden, t_hidden in zip(s_hiddens_selected, t_hiddens_selected):
            # if args.model_type == 'opt':
            #     s_decoder_proj = student.module.model.model.decoder.project_out
            #     if s_decoder_proj is not None:
            #         s_hidden = s_decoder_proj(s_hidden)

            #     t_decoder_proj = teacher.model.decoder.project_out
            #     if t_decoder_proj is not None:
            #         t_hidden = t_decoder_proj(t_hidden)

            s_hidden_logits = student.module.lm_head(s_hidden)
            t_hidden_logits = teacher.lm_head(t_hidden)
            # traj_loss += forward_kl(s_hidden_logits, t_hidden_logits, no_model_batch)
            traj_loss += soft_label_distill_loss(s_hidden_logits, t_hidden_logits, mask)

            s_hidden_logs = F.log_softmax(s_hidden_logits, dim=-1)
            t_hidden_logs = F.log_softmax(t_hidden_logits, dim=-1)

            if i > 0:
                delta_hidden_student = s_hidden_logs - pre_s_hidden_logs
                delta_hidden_teacher = t_hidden_logs - pre_t_hidden_logs
                cos_sim = F.cosine_similarity(delta_hidden_student, delta_hidden_teacher, dim=-1, eps=1e-5)
                cos_sim_loss = 1 - cos_sim
                cos_sim_loss = (cos_sim_loss * mask).sum() / mask.sum()

                der_loss +=  cos_sim_loss

            pre_s_hidden_logs, pre_t_hidden_logs = s_hidden_logs, t_hidden_logs

            i += 1

        return traj_loss / i +  der_loss / (i - 1)

def get_csd_loss(logits, teacher_logits, no_model_batch, mode="SS"):
    student_probs = F.softmax(logits, dim=-1)
    teacher_probs = F.softmax(logits, dim=-1)
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits) | torch.isneginf(logits) | torch.isneginf(teacher_logits)
    
    assert type(mode) == str and len(mode) == 2, "wrong mode format"
    def get_weight_func(mode):
        if mode == "S":
            return torch.clone(student_probs).detach()
        if mode == "T":
            return torch.clone(teacher_probs).detach()
        if mode == "U":
            return torch.ones(student_probs.shape) / student_probs.shape[-1]
        raise Exception("unsupported mode")

    w1 = get_weight_func(mode[0]).to(device=logits.device)
    w2 = get_weight_func(mode[1]).to(device=logits.device)
    
    s_t_diff = torch.masked_fill(logits - teacher_logits, inf_mask, 0)
    w2_diff = torch.masked_fill(w2 * s_t_diff, inf_mask, 0)
    w1_diff = torch.masked_fill(w1 * s_t_diff, inf_mask, 0)
    
    w_grad = torch.masked_fill(w1 * (s_t_diff - torch.sum(w2_diff, dim=-1, keepdim=True)), inf_mask, 0)
    w_grad += torch.masked_fill(w2 * (s_t_diff - torch.sum(w1_diff, dim=-1, keepdim=True)), inf_mask, 0)
    csd_loss_per_token = torch.masked_fill(w_grad.detach() * logits / 2, inf_mask, 0).sum(dim=-1)
    
    mask = (no_model_batch["label"] != -100).int()
    csd_loss = torch.sum(csd_loss_per_token.view(-1) * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return csd_loss

def ab_div(logits, teacher_logits, no_model_batch, alpha, beta):
    """Calculate D^{(alpha, beta)} divergence."""
    log_p = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    log_q = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    eps = 1e-8

    if abs(alpha) < eps and abs(beta) < eps:
        # Case 1: alpha = 0, beta = 0
        divergence = 0.5 * torch.sum((log_q - log_p).pow(2), dim=-1)

    elif abs(alpha) < eps:
        # Case 2: alpha = 0, beta != 0
        safe_log_ratio_div_beta = torch.where(
            torch.isfinite(log_q - log_p), log_q - log_p, 0.0
        )
        divergence = torch.sum(
            torch.exp(beta * log_q) * (beta * safe_log_ratio_div_beta - 1) + torch.exp(beta * log_p),
            dim=-1
        ) / (beta ** 2)

    elif abs(beta) < eps:
        # Case 3: beta = 0, alpha != 0
        safe_log_ratio_div_alpha = torch.where(
            torch.isfinite(log_p - log_q), log_p - log_q, 0.0
        )
        divergence = torch.sum(
            torch.exp(alpha * log_p) * (alpha * safe_log_ratio_div_alpha - 1) + torch.exp(alpha * log_q),
            dim=-1
        ) / (alpha ** 2)

    elif abs(alpha + beta) < eps:
        # Case 4: alpha + beta = 0
        safe_log_r = torch.where(torch.isfinite(log_q - log_p), log_q - log_p, 0.0)
        divergence = torch.sum(
            alpha * safe_log_r + torch.exp(-alpha * safe_log_r) - 1,
            dim=-1
        ) / (alpha ** 2)

    else:
        # General case
        apb = alpha + beta
        term1 = torch.exp(torch.logsumexp(alpha * log_p + beta * log_q, dim=-1))
        term2 = (alpha / apb) * torch.exp(torch.logsumexp(apb * log_p, dim=-1))
        term3 = (beta / apb) * torch.exp(torch.logsumexp(apb * log_q, dim=-1))
        divergence = - (term1 - term2 - term3) / (alpha * beta)
        
    mask = (no_model_batch["label"] != -100).float()
    
    # Replace any NaN/inf values in the divergence with 0.0 for safe masking
    safe_divergence = torch.where(torch.isfinite(divergence), divergence, 0.0)
    
    # Apply the mask and compute the final mean loss
    masked_sum = (safe_divergence * mask).sum()
    mask_sum = mask.sum()
    
    # Avoid division by zero if the mask is empty
    loss = masked_sum / mask_sum if mask_sum > 0 else masked_sum
    return loss
