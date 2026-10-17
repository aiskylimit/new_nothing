"""Segment-Selective SFT (Wang, Liu & Ren, ICLR 2026), as run by the SegmentSelectiveSFT fork.

    (sgl.data.s1k) s1K-1.1 -> {question, solution, answer, segments}      (prepare_s1k.py + segment_split.py)
    attribution  Integrated Gradients from every token to the answer        (Attribution/grad_analyze.py)
    select       important segments: strength top-k + coherence filter      (Attribution/get_important_segments.py)
    build        tokenized {input_ids, loss_mask} exactly as train_mask.py  (SelectiveSFT/train_mask.py)

Run with `ssft run configs/ssft/<experiment>.yaml`. Training goes through sgl.training.train
(native) or the untouched references/SegmentSelectiveSFT/SelectiveSFT/train_mask.py (reference).
"""
