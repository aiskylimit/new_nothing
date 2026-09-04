import torch
import torch.nn as nn 
import torch.distributed as dist
import torch.nn.functional as F
from src.criterions.utils import count_clean_text_tokens, get_hidden_text, get_hidden_text_vision, pooling
import random
import os


class TalasJepa(nn.Module):
    def __init__(self, args):
        super(TalasJepa, self).__init__()
        self.args = args
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0
        self.kd_weight = args.kd_weight
        self.step = 0
        self.warm_up_projector = 1000
    
    def _dist_gather_tensor(self, t: torch.Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def cosine_loss(self, student_embeddings, teacher_embeddings):
        cos_sim = F.cosine_similarity(student_embeddings, teacher_embeddings, dim=-1)
        cos_sim_loss = 1 - cos_sim
        return cos_sim_loss.mean()

    def structure_loss(self, student_embeddings, teacher_embeddings):
        student_embeddings = F.normalize(student_embeddings, p=2, dim=-1)
        teacher_embeddings = F.normalize(teacher_embeddings, p=2, dim=-1)

        student_similarity = student_embeddings @ student_embeddings.transpose(-1, -2)
        teacher_similarity = teacher_embeddings @ teacher_embeddings.transpose(-1, -2)

        loss = F.mse_loss(student_similarity, teacher_similarity)

        return loss

    def distillcse_kd_loss(self, S1, S2, T1, T2, tau=0.05,):
        """
        Distill teacher similarity distribution over in-batch negatives.

        Student and teacher dimensions do not need to match because
        distillation is applied to pairwise similarity matrices.
        """
        S1 = F.normalize(S1.float(), p=2, dim=-1)

        S2 = F.normalize(S2.float(), p=2, dim=-1)

        T1 = F.normalize(T1.float(), p=2, dim=-1)

        T2 = F.normalize(T2.float(), p=2, dim=-1,)

        s_logits = (S1 @ S2.transpose(0, 1)) / tau

        t_logits = (T1 @ T2.transpose(0, 1)) / tau

        # Positive query-passage pairs are on the diagonal.
        # DistillCSE KD here focuses on the negative distribution.
        mask = torch.eye(s_logits.size(0), device=s_logits.device, dtype=torch.bool,)

        s_logits = s_logits.masked_fill(mask, torch.finfo(s_logits.dtype).min,)

        t_logits = t_logits.masked_fill( mask,torch.finfo(t_logits.dtype).min,)

        teacher_probs = F.softmax(t_logits,dim=1,).detach()

        student_log_probs = F.log_softmax(s_logits,dim=1,)

        return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean",)
    
    def sigreg(self, x: torch.Tensor, num_slices: int = 128) -> torch.Tensor:
        device = x.device
        # =====================================================
        # 1. Random projection seed
        #
        # Chỉ rank 0 sinh seed.
        # Sau đó broadcast để tất cả GPU dùng cùng seed.
        # =====================================================
        if self.process_rank == 0:
            projection_seed = random.randint(0, 2**63 - 1)
        else:
            projection_seed = 0

        if self.world_size > 1:
            seed_tensor = torch.tensor(projection_seed, dtype=torch.int64, device=device,)
            dist.broadcast(seed_tensor, src=0)
            projection_seed = seed_tensor.item()

        # =====================================================
        # 2. Local generator
        #
        # Không làm thay đổi global torch RNG state.
        # Tất cả GPU có cùng projection_seed -> cùng A.
        # =====================================================
        g = torch.Generator(device=device)
        g.manual_seed(projection_seed)

        A = torch.randn(x.size(1), num_slices, generator=g,  device=device, dtype=x.dtype,)

        A = A / A.norm(p=2, dim=0, keepdim=True, ).clamp_min(1e-12)

        # =====================================================
        # 3. Epps-Pulley statistic
        # =====================================================
        t = torch.linspace(-5, 5, 17, device=device, dtype=x.dtype,)

        exp_f = torch.exp(-0.5 * t.square())

        # x:   [N, K]
        # A:   [K, M]
        # x@A: [N, M]
        # x_t: [N, M, T]
        x_t = (x @ A).unsqueeze(-1) * t

        # [M, T]
        ecf = torch.exp(1j * x_t).mean(dim=0)

        # =====================================================
        # 4. Aggregate across GPUs
        # =====================================================
        if self.world_size > 1:
            dist.all_reduce(ecf, op=dist.ReduceOp.SUM,)
            ecf = ecf / self.world_size

        # =====================================================
        # 5. Weighted L2 distance
        # =====================================================
        err = ((ecf - exp_f).abs().square().mul(exp_f))

        global_batch_size = x.size(0) * self.world_size

        sigreg_per_slice = (torch.trapezoid(err, t, dim=1,) * global_batch_size)

        return sigreg_per_slice.mean()

    def sigreg_orthogonal_per_sample(self, current_tokens: torch.Tensor, 
                                     next_layer_tokens: torch.Tensor, 
                                     u_anchor: torch.Tensor,
                                     num_slices: int = 128) -> tuple:
        """
        Tính Cascade KD Alignment và Orthogonal SIGReg per sample.
        current_tokens: [B, N_tokens, Dim]
        next_layer_tokens: [B, N_tokens, Dim]
        """
        device = current_tokens.device
        dtype = current_tokens.dtype
        B, N_tokens, D = current_tokens.shape

        # =====================================================
        # 1. Cascade Alignment (Layer l aligns to Layer l+1)
        # =====================================================
        # current_mean = current_tokens.mean(dim=1)     # [B, D]
        # next_layer_mean = next_layer_tokens.mean(dim=1) # [B, D]
        
        # Loss KD nối tiếp (Mean Token Alignment)
        # align_loss = F.mse_loss(current_mean, next_layer_mean.detach())
        # align_loss = self.structure_loss(current_mean, next_layer_mean.detach())
        align_loss = 0.0

        # =====================================================
        # 2. Phân rã không gian (Orthogonal Projection)
        # =====================================================
        # u = F.normalize(u_anchor.detach(), p=2, dim=-1, eps=1e-8) # [B, D]
        u = u_anchor
        u_expanded = u.unsqueeze(1) # [B, 1, D]

        dot_product = torch.sum(current_tokens * u_expanded, dim=-1, keepdim=True) # [B, N_tokens, 1]
        z_parallel = dot_product * u_expanded # [B, N_tokens, D]
        z_perp = current_tokens - z_parallel  # [B, N_tokens, D]

        # Khôi phục variance về ~1 cho phần dư để phù hợp với hàm mũ của SIGReg
        z_perp_normalized = F.layer_norm(z_perp, (D,))

        # =====================================================
        # 3. Tính SIGReg trên phần dư (Instance-wise / Không cần all_reduce)
        # =====================================================
        if getattr(self, 'process_rank', 0) == 0:
            projection_seed = random.randint(0, 2**63 - 1)
        else:
            projection_seed = 0

        if getattr(self, 'world_size', 1) > 1:
            seed_tensor = torch.tensor(projection_seed, dtype=torch.int64, device=device)
            dist.broadcast(seed_tensor, src=0)
            projection_seed = seed_tensor.item()

        g = torch.Generator(device=device)
        g.manual_seed(projection_seed)

        A = torch.randn(D, num_slices, generator=g, device=device, dtype=dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)

        t = torch.linspace(-5, 5, 17, device=device, dtype=dtype)
        exp_f = torch.exp(-0.5 * t.square()) # [17]

        # Chiếu dữ liệu: [B, N_tokens, D] @ [D, num_slices] -> [B, N_tokens, num_slices]
        x_proj = z_perp_normalized @ A
        
        # [B, N_tokens, num_slices, 1] * [17] -> [B, N_tokens, num_slices, 17]
        x_t = x_proj.unsqueeze(-1) * t

        # Tính Characteristic Function cho TỪNG SAMPLE (mean theo dim=1 tương ứng N_tokens)
        ecf = torch.exp(1j * x_t).mean(dim=1) # [B, num_slices, 17]

        # Khoảng cách L2
        err = (ecf - exp_f).abs().square().mul(exp_f) # [B, num_slices, 17]
        
        # Tích phân Epps-Pulley
        T_stat = torch.trapezoid(err, t, dim=-1) * N_tokens # [B, num_slices]

        # Lấy trung bình qua các lát cắt và toàn batch
        sigreg_loss = T_stat.mean()

        return align_loss, sigreg_loss

    def _compute_vision_distill(self, student_hidden_states, image_features, 
                                teacher_img_reps, text_token_counts, 
                                attention_mask, t2s_projector):

        if teacher_img_reps is None or image_features is None:
            return 0.0, 0.0, 0.0

        batch_size = attention_mask.size(0)
        num_layers = len(student_hidden_states)
        start_layer = int(num_layers * 0.4)
        end_layer = num_layers
        
        collect_end_layer = min(end_layer + 1, num_layers)
        stu_img_tokens_mid = {l: [] for l in range(start_layer, collect_end_layer)}
        stu_img_reps = []
        
        cur_idx_img = 0
        
        for i in range(batch_size):
            if cur_idx_img < len(image_features):
                img_feat = image_features[cur_idx_img]
                num_vision_token = img_feat.size(0)
                
                # Trích xuất biểu diễn layer cuối cùng
                _, last_stu_img_hidden_state = get_hidden_text_vision(
                    student_hidden_states[-1][i],
                    text_token_counts[i].item(),
                    num_vision_token,
                    attention_mask[i]
                )
                stu_img_reps.append(last_stu_img_hidden_state.mean(dim=0))

                # Trích xuất các layer giữa
                for l in range(start_layer, collect_end_layer):
                    _, mid_stu_img_hidden_state = get_hidden_text_vision(
                        student_hidden_states[l][i],
                        text_token_counts[i].item(),
                        num_vision_token,
                        attention_mask[i]
                    )
                    stu_img_tokens_mid[l].append(mid_stu_img_hidden_state)

                cur_idx_img += 1

        if len(stu_img_reps) == 0:
            return 0.0, 0.0, 0.0

        stu_img_reps = torch.stack(stu_img_reps, dim=0)

        sigreg_total = 0.0
        cascade_align_loss = 0.0
        sigreg_final = 0.0
        cascade_align_final = 0.0
        actual_end_layer = min(end_layer, collect_end_layer - 1)

        if self.step > self.warm_up_projector:
            # Lấy Anchor từ layer cuối cùng (chứa ngữ nghĩa cao nhất)
            u_top = F.normalize(stu_img_reps.detach(), p=2, dim=-1) 

            for l in range(start_layer, actual_end_layer):
                current_layer_tokens = torch.stack(stu_img_tokens_mid[l], dim=0) # [B, N, D]
                next_layer_tokens = torch.stack(stu_img_tokens_mid[l+1], dim=0)  # [B, N, D]
                
                l_align, l_sigreg = self.sigreg_orthogonal_per_sample(
                    current_layer_tokens, 
                    next_layer_tokens, 
                    u_anchor=u_top
                )
                cascade_align_loss += l_align
                sigreg_total += l_sigreg
        
            num_mid_layers = max(actual_end_layer - start_layer, 1)
            
            sigreg_final = sigreg_total / num_mid_layers
            cascade_align_final = cascade_align_loss / num_mid_layers

            vision_align_loss = self.cosine_loss(stu_img_reps, t2s_projector(teacher_img_reps))

        else:
            vision_align_loss = self.cosine_loss(stu_img_reps.detach(), t2s_projector(teacher_img_reps))

        return vision_align_loss, cascade_align_final, sigreg_final

    def forward(self, model_wrapper, input_data):
        self.step += 1

        student_model = model_wrapper.model
        student_processor = model_wrapper.get_processor()
        student_tokenizer = student_processor.tokenizer
        projectors = model_wrapper.projectors        

        student_qry_input = input_data['qry']
        student_pos_input = input_data['pos']
        
        batch_size = student_qry_input['input_ids'].size(0)

        student_qry_output = student_model.encode_input(student_qry_input)
        student_pos_output = student_model.encode_input(student_pos_input)
        student_qry_reps, student_qry_image_features, student_qry_attention, student_qry_hidden_states = student_qry_output
        student_pos_reps, student_pos_image_features, student_pos_attention, student_pos_hidden_states = student_pos_output

        device = student_qry_reps.device
        dtype = student_qry_reps.dtype

        teacher_qry, teacher_pos = input_data["teacher_qry_caches"], input_data["teacher_pos_caches"]

        teacher_qry_reps = torch.stack([rep['rep'] for rep in teacher_qry], dim=0).to(device, dtype=dtype)
        teacher_pos_reps = torch.stack([rep['rep'] for rep in teacher_pos], dim=0).to(device, dtype=dtype)

        tea_img_qry_reps = torch.stack([rep['mean_last_img_token'] for rep in teacher_qry], 
                                       dim=0,).to(device, dtype=dtype) if teacher_qry[0]['mean_last_img_token'] is not None else None
        tea_img_pos_reps = torch.stack([rep['mean_last_img_token'] for rep in teacher_pos], 
                                       dim=0,).to(device, dtype=dtype) if teacher_pos[0]['mean_last_img_token'] is not None else None
        
        if getattr(self, 'world_size', 1) > 1:
            all_student_qry_reps = self._dist_gather_tensor(student_qry_reps)
            all_student_pos_reps = self._dist_gather_tensor(student_pos_reps)
            all_teacher_qry_reps = self._dist_gather_tensor(teacher_qry_reps)
            all_teacher_pos_reps = self._dist_gather_tensor(teacher_pos_reps)
        else:
            all_student_qry_reps = student_qry_reps
            all_student_pos_reps = student_pos_reps
            all_teacher_qry_reps = teacher_qry_reps
            all_teacher_pos_reps = teacher_pos_reps
            
        scores = student_model.compute_similarity(all_student_qry_reps, all_student_pos_reps)
        scores = scores.view(all_student_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_student_qry_reps.size(0) // all_student_pos_reps.size(0))
        contrastive_loss = nn.CrossEntropyLoss()(scores / model_wrapper.temperature, target)

        kd_simcse = 0.0
        last_stu_qry_hidden_state = pooling(student_qry_hidden_states[-1], 
                                            student_qry_input['attention_mask'], 
                                            mode='eos',
                                            normalize=True)
        last_stu_pos_hidden_state = pooling(student_pos_hidden_states[-1], 
                                            student_pos_input['attention_mask'], 
                                            mode='eos',
                                            normalize=True)
        
        kd_simcse += self.distillcse_kd_loss(last_stu_qry_hidden_state, last_stu_pos_hidden_state, 
                                            teacher_qry_reps, teacher_pos_reps)

        ##################################
        student_special_ids = torch.tensor(
            list(set(list(student_tokenizer.added_tokens_encoder.values()) + student_tokenizer.all_special_ids) 
                 - set([student_tokenizer.eos_token_id])),
            device=student_qry_input['input_ids'].device,
            dtype=torch.long
        )

        num_student_text_qry_tokens = count_clean_text_tokens(student_qry_input, student_special_ids)
        num_student_text_pos_tokens = count_clean_text_tokens(student_pos_input, student_special_ids)

        vision_loss = torch.zeros_like(contrastive_loss)
        SIGReg = torch.zeros_like(contrastive_loss)
        
        if tea_img_qry_reps is not None:
            qry_align, qry_cascade, qry_sigreg = self._compute_vision_distill(
                student_hidden_states=student_qry_hidden_states,
                image_features=student_qry_image_features,
                teacher_img_reps=tea_img_qry_reps,
                text_token_counts=num_student_text_qry_tokens,
                attention_mask=student_qry_input['attention_mask'],
                t2s_projector=projectors['t2s']
            )
            vision_loss += qry_align
            SIGReg += qry_sigreg

        if tea_img_pos_reps is not None:
            pos_align, pos_cascade, pos_sigreg = self._compute_vision_distill(
                student_hidden_states=student_pos_hidden_states,
                image_features=student_pos_image_features,
                teacher_img_reps=tea_img_pos_reps,
                text_token_counts=num_student_text_pos_tokens,
                attention_mask=student_pos_input['attention_mask'],
                t2s_projector=projectors['t2s']
            )
            vision_loss += pos_align
            SIGReg += pos_sigreg

        # --- TRUNG BÌNH CHUNG ---
        if tea_img_qry_reps is not None and tea_img_pos_reps is not None:
            vision_loss = vision_loss / 2
            SIGReg = SIGReg / 2

        loss_distill = torch.zeros_like(contrastive_loss)
        if self.args.use_distill_cse_loss:
            loss_distill += kd_simcse
        if self.args.use_distill_vison_loss:
            loss_distill += vision_loss 

        loss = contrastive_loss 
        if self.args.use_distill_loss:
            loss = loss + self.kd_weight * loss_distill
        if self.args.use_sigreg_loss:
            loss = loss + self.args.sigreg_weight * SIGReg

        return {
            'loss': loss,
            'contrastive_loss': contrastive_loss,
            'kd_loss': loss_distill,
            'kd_loss_simcse': kd_simcse,
            'sigreg_loss': SIGReg
        }
