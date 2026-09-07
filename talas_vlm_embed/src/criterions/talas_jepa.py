import torch
import torch.nn as nn 
import torch.distributed as dist
import torch.nn.functional as F
from src.criterions.utils import count_clean_text_tokens, get_hidden_text, get_hidden_text_vision, pooling
import random
import os
import math


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

        self.counter = 0
        self.warm_up_sigreg = 0
    
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

    def distillcse_kd_loss(self, S1, S2, T1, T2, tau=0.04):
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

    def sigreg_sinkhorn(self, z: torch.Tensor, concept_queries,
                        tau: float = 0.05, n_iters: int = 3):
        """
        z shape: [B, N, D] - Toàn bộ batch ảnh với N tokens mỗi ảnh
        """
        B, N, D = z.shape
        device, dtype = z.device, z.dtype
        
        # # ==========================================
        # # 1. DIVERSITY LOSS: Ép K mỏ neo phải phân tách
        # # ==========================================
        queries_norm = F.normalize(concept_queries, p=2, dim=-1)
        # query_sim_matrix = queries_norm @ queries_norm.T # [K, K]
        # loss_diversity = F.mse_loss(query_sim_matrix, torch.eye(self.K, device=device, dtype=dtype))
        
        # ==========================================
        # 2. BATCH-WISE SINKHORN-KNOPP
        # ==========================================
        z_norm = F.normalize(z, p=2, dim=-1) # [B, N, D]
        
        cost_matrix = 1.0 - torch.einsum('kd,bnd->bkn', queries_norm, z_norm)
        log_Q = -cost_matrix / tau
        
        # Chạy Sinkhorn ngầm
        with torch.no_grad():
            for _ in range(n_iters - 1):
                log_Q = log_Q - torch.logsumexp(log_Q, dim=1, keepdim=True) # Cân bằng K
                log_Q = log_Q - torch.logsumexp(log_Q, dim=2, keepdim=True) # Cân bằng N
                
        # Vòng cuối CÓ gradient (BẮT BUỘC KẾT THÚC BẰNG DIM=2)
        log_Q = log_Q - torch.logsumexp(log_Q, dim=1, keepdim=True) 
        log_Q = log_Q - torch.logsumexp(log_Q, dim=2, keepdim=True) 
        
        affinity = torch.exp(log_Q) # [B, K, N]
        
        # Rút ra K centroids: [B, K, D]
        z_centroids = torch.bmm(affinity, z) 
        
        # ==========================================
        # 3. CHUẨN BỊ KHÔNG GIAN BẰNG RMSNorm
        # ==========================================
        z_k_concepts = z_centroids.transpose(0, 1) 
        z_normed = z_k_concepts / z_k_concepts.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12) * math.sqrt(D)
            
        # ==========================================
        # 4. BATCH-WISE SIGREG TRÊN KHÔNG GIAN TINH KHIẾT
        # ==========================================
        A = torch.randn(D, self.num_slices, device=device, dtype=dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)
        t = torch.linspace(-5, 5, 17, device=device, dtype=dtype)
        exp_f = torch.exp(-0.5 * t.square())
        
        x_proj = z_normed @ A                  # [K, B, num_slices]
        x_t = x_proj.unsqueeze(-1) * t         # [K, B, num_slices, 17]
        
        # TÍNH ECF: Lấy trung bình dọc theo BATCH (dim=1)
        ecf = torch.exp(1j * x_t).mean(dim=1)  # [K, num_slices, 17]
        
        err = (ecf - exp_f).abs().square().mul(exp_f)
        loss_sigreg = torch.trapezoid(err, t, dim=-1).mean(dim=-1) * B 
        
        total_loss = loss_sigreg.mean()
        # total_loss += 0.1 * loss_diversity 
        
        return total_loss


    def _compute_modality_distill(self, student_hidden_states, image_features, 
                                  text_token_counts, attention_mask, concept_queries):
        """
        Hàm này chỉ còn nhiệm vụ trích xuất text và vision representations 
        của student, cùng với việc tính toán SIGReg loss.
        """
        k_layers = self.args.num_layers
        batch_size = attention_mask.size(0)
        last_layer_idx = len(student_hidden_states) - 15
        
        start_sigreg_layer = max(0, last_layer_idx - k_layers)
        
        stu_img_tokens = {l: [] for l in range(start_sigreg_layer, last_layer_idx + 1)}
        stu_text_reps = []
        
        cur_idx_img = 0
        for i in range(batch_size):
            num_vision_token = 0
            if image_features is not None and cur_idx_img < len(image_features):
                num_vision_token = image_features[cur_idx_img].size(0)
                cur_idx_img += 1
            
            text_last_hidden, img_last_hidden = get_hidden_text_vision(
                student_hidden_states[last_layer_idx][i],
                text_token_counts[i].item(),
                num_vision_token,
                attention_mask[i]
            )
            stu_text_reps.append(text_last_hidden.mean(dim=0))
            
            if num_vision_token > 0:
                for l in range(start_sigreg_layer, last_layer_idx + 1):
                    _, img_hidden = get_hidden_text_vision(
                        student_hidden_states[l][i],
                        text_token_counts[i].item(),
                        num_vision_token,
                        attention_mask[i]
                    )
                    stu_img_tokens[l].append(img_hidden)

        # 1. Gom representations của Text
        stacked_stu_text_reps = torch.stack(stu_text_reps, dim=0)

        # 2. Gom representations của Vision và tính SIGReg
        stu_img_final_reps = None
        sigreg_final = 0.0
        
        if len(stu_img_tokens[last_layer_idx]) > 0:
            stu_img_final_reps = torch.stack([x.mean(dim=0) for x in stu_img_tokens[last_layer_idx]], dim=0) 

            warmup_factor = min(1.0, self.counter / max(1, self.warm_up_sigreg))
            total_sigreg = 0.0
            
            # Duyệt qua các layer từ L-k đến L-1
            for l in range(start_sigreg_layer, last_layer_idx):
                # MỎ NEO LÀ MEAN CỦA LAYER L+1 (Detach để an toàn)
                # anchors_l_plus_1 = [x.mean(dim=0).detach() for x in stu_img_tokens[l+1]]
                
                # # Gọi SIGReg với mỏ neo truyền vào
                # layer_sigreg = self.sigreg_orthogonal_per_sample(
                #     tokens_list=stu_img_tokens[l],
                #     # anchors_list=anchors_l_plus_1
                # )

                layer_sigreg = 0.0
                for tokens in stu_img_tokens[l]:
                    layer_sigreg += self.sigreg_sinkhorn(tokens, concept_queries)

                total_sigreg += layer_sigreg / len(stu_img_tokens[l])
                
            sigreg_final = warmup_factor * (total_sigreg / max(1, k_layers))

        return stacked_stu_text_reps, stu_img_final_reps, sigreg_final
    
    def forward(self, model_wrapper, input_data):
        student_model = model_wrapper.model
        student_processor = model_wrapper.get_processor()
        student_tokenizer = student_processor.tokenizer
        concept_queries = model_wrapper.concept_queries      

        student_qry_input = input_data['qry']
        student_pos_input = input_data['pos']
        
        batch_size = student_qry_input['input_ids'].size(0)
        self.counter += batch_size

        student_qry_output = student_model.encode_input(student_qry_input)
        student_pos_output = student_model.encode_input(student_pos_input)
        student_qry_reps, student_qry_image_features, student_qry_attention, student_qry_hidden_states = student_qry_output
        student_pos_reps, student_pos_image_features, student_pos_attention, student_pos_hidden_states = student_pos_output

        device = student_qry_reps.device
        dtype = student_qry_reps.dtype

        teacher_qry, teacher_pos = input_data["teacher_qry_caches"], input_data["teacher_pos_caches"]

        teacher_qry_reps = torch.stack([rep['rep'] for rep in teacher_qry], dim=0).to(device, dtype=dtype)
        teacher_pos_reps = torch.stack([rep['rep'] for rep in teacher_pos], dim=0).to(device, dtype=dtype)

        tea_img_qry_reps = torch.stack([rep['mean_last_img_token'] for rep in teacher_qry], dim=0).to(device, dtype=dtype) if teacher_qry[0]['mean_last_img_token'] is not None else None
        tea_img_pos_reps = torch.stack([rep['mean_last_img_token'] for rep in teacher_pos], dim=0).to(device, dtype=dtype) if teacher_pos[0]['mean_last_img_token'] is not None else None

        tea_text_qry_reps = torch.stack([rep['mean_last_text_token'] for rep in teacher_qry], dim=0).to(device, dtype=dtype) if teacher_qry[0]['mean_last_text_token'] is not None else None
        tea_text_pos_reps = torch.stack([rep['mean_last_text_token'] for rep in teacher_pos], dim=0).to(device, dtype=dtype) if teacher_pos[0]['mean_last_text_token'] is not None else None
        
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
                                            mode='eos', normalize=True)
        last_stu_pos_hidden_state = pooling(student_pos_hidden_states[-1], 
                                            student_pos_input['attention_mask'], 
                                            mode='eos', normalize=True)
        
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

        # Trích xuất Representations từ QRY
        qry_stu_txt, qry_stu_img, qry_sigreg = self._compute_modality_distill(
            student_hidden_states=student_qry_hidden_states, 
            image_features=student_qry_image_features,
            text_token_counts=num_student_text_qry_tokens, 
            attention_mask=student_qry_input['attention_mask'], 
            concept_queries=concept_queries
        )

        # Trích xuất Representations từ POS
        pos_stu_txt, pos_stu_img, pos_sigreg = self._compute_modality_distill(
            student_hidden_states=student_pos_hidden_states, 
            image_features=student_pos_image_features,
            text_token_counts=num_student_text_pos_tokens, 
            attention_mask=student_pos_input['attention_mask'], 
            concept_queries=concept_queries
        )

        stu_modality_features = []
        tea_modality_features = []
        SIGReg = torch.zeros_like(contrastive_loss)
        num_sigreg_components = 0

        if tea_text_qry_reps is not None:
            stu_modality_features.append(qry_stu_txt)
            tea_modality_features.append(tea_text_qry_reps)
        if tea_text_pos_reps is not None:
            stu_modality_features.append(pos_stu_txt)
            tea_modality_features.append(tea_text_pos_reps)

        if tea_img_qry_reps is not None and qry_stu_img is not None:
            stu_modality_features.append(qry_stu_img)
            tea_modality_features.append(tea_img_qry_reps)
            SIGReg += qry_sigreg
            num_sigreg_components += 1

        if tea_img_pos_reps is not None and pos_stu_img is not None:
            stu_modality_features.append(pos_stu_img)
            tea_modality_features.append(tea_img_pos_reps)
            SIGReg += pos_sigreg
            num_sigreg_components += 1

        if num_sigreg_components > 0:
            SIGReg = SIGReg / num_sigreg_components

        modality_loss = torch.zeros_like(contrastive_loss)
        if len(stu_modality_features) > 0:
            all_stu_modality = torch.cat(stu_modality_features, dim=0)
            all_tea_modality = torch.cat(tea_modality_features, dim=0)
            modality_loss = self.structure_loss(all_stu_modality, all_tea_modality)

        # ==============================================================

        loss_distill = torch.zeros_like(contrastive_loss)
        if self.args.use_distill_cse_loss:
            loss_distill += kd_simcse
            
        if self.args.use_distill_vison_loss:
            loss_distill += modality_loss

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