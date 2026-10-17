"""P-ALIGN: Long-Chain Reasoning Distillation via Adaptive Prefix Alignment (arXiv 2601.10064).

    (sgl.data.s1k)  s1K-1.1 -> {question, solution (teacher long CoT), answer}
    truncate        binary search for the shortest teacher prefix the student judges sufficient
    align           the student continues from that prefix
    build           <Begin_of_Prefix>prefix<End_of_Prefix>\\ncontinuation, kept when the answer is right

The result is P-ALIGN's training file (alpaca rows), trained with uniform NLL. Run with
`palign run configs/palign/<experiment>.yaml`.
"""
