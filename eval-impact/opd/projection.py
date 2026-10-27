"""IMPACT embedding-overlap ridge initializer, adapted from code/impact_utils.py."""
import torch

def _normalize_tokenizer_vocab(vocab):
    """Unify GPT-2 (Ġ) and SentencePiece (▁) space markers for overlap counting."""
    return {k.replace("Ġ", "▁"): v for k, v in vocab.items()}



def compute_vocab_projection(teacher_model, student_model, teacher_tokenizer,
                             student_tokenizer, lambda_reg=1.0, max_overlap=None):
    """Solve for W_{T->S} via ridge regression on overlapping vocabulary embeddings.

    W = (E_T^T E_T + lambda * I)^{-1} E_T^T E_S
    Returns W of shape [d_T, d_S].

    If max_overlap is set and smaller than |V_overlap|, keep only the
    max_overlap overlapping tokens with the lowest student token ids
    (BPE frequency proxy: lower id ≈ more frequent).
    """
    stu_norm = _normalize_tokenizer_vocab(student_tokenizer.get_vocab())
    tea_norm = _normalize_tokenizer_vocab(teacher_tokenizer.get_vocab())
    overlap = sorted(set(stu_norm.keys()) & set(tea_norm.keys()))

    if len(overlap) == 0:
        d_T = teacher_model.config.hidden_size
        d_S = student_model.config.hidden_size
        return torch.zeros(d_T, d_S)

    n_full = len(overlap)
    if max_overlap is not None and int(max_overlap) > 0 and n_full > int(max_overlap):
        # Rank by student token id (lower ≈ more frequent in BPE/Unigram vocabs).
        overlap = sorted(overlap, key=lambda t: stu_norm[t])[: int(max_overlap)]

    stu_ids = [stu_norm[t] for t in overlap]
    tea_ids = [tea_norm[t] for t in overlap]

    stu_emb = student_model.get_input_embeddings()
    tea_emb = teacher_model.get_input_embeddings()

    E_S = stu_emb.weight.data[stu_ids].float()  # [V_overlap, d_S]
    E_T = tea_emb.weight.data[tea_ids].float()  # [V_overlap, d_T]

    d_T = E_T.size(1)
    A = E_T.T @ E_T + lambda_reg * torch.eye(d_T, device=E_T.device)
    B = E_T.T @ E_S
    W = torch.linalg.solve(A, B)  # [d_T, d_S]

    return W.detach()

