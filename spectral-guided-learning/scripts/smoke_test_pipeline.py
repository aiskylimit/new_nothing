"""CPU end-to-end smoke test of the Spectral-guided Learning pipeline.

Runs segmentation -> gradient capture -> SVD -> step selection -> mask -> a real
training step, using the real Qwen3 tokenizer with a tiny randomly initialized model.
Validates the plumbing without a GPU; it says nothing about result quality.

    python scripts/smoke_test_pipeline.py
"""

import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_collator import MaskedSFTCollator  # noqa: E402
from gradient_utils import capture_sequence_gradients  # noqa: E402
from masked_loss import masked_cross_entropy  # noqa: E402
from segmentation import segment_response_token_spans  # noqa: E402
from spectral_utils import analyze_gradient_matrix  # noqa: E402
from step_selection import build_loss_mask, select_steps_by_energy, selection_stats  # noqa: E402

TOKENIZER_NAME = "Qwen/Qwen3-0.6B-Base"

SAMPLES = [
    (
        "In triangle ABC, side AC is the largest. Find angle ABC.",
        "We need to solve this geometry problem. In triangle ABC, side AC is the largest, "
        "and points M and N lie on AC with AM = AB and CN = CB.\n\n"
        "Wait, let me double check that. Hmm, actually maybe I should reconsider the setup "
        "here, since something feels off about it.\n\n"
        "Since AM = AB, triangle ABM is isosceles with apex at B, so angle AMB equals "
        "angle ABM. The same argument applies to triangle CBN.\n\n"
        "That could be interpreted as angle NBM = angle ABC / 3, which is what we want.\n\n"
        "Therefore angle ABC = 108 degrees. The answer is \\boxed{108}.",
    ),
    (
        "Compute the sum of the first 10 positive integers.",
        "Let me apply the standard formula for an arithmetic series.\n\n"
        "The sum of the first n positive integers is n(n+1)/2, a well known identity.\n\n"
        "Hmm, wait. Let me just double-check by adding a few terms manually first.\n\n"
        "Substituting n = 10 gives 10 * 11 / 2 = 55, so the answer is \\boxed{55}.",
    ),
]


def build_tiny_model(vocab_size: int):
    """A 2-layer Qwen3 with the real vocabulary — small enough to run on CPU."""
    config = AutoConfig.for_model(
        "qwen3",
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(config).eval().float()


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = build_tiny_model(len(tokenizer))
    print(f"tokenizer vocab={len(tokenizer)}  tiny model params={sum(p.numel() for p in model.parameters()):,}")

    examples = []
    for problem, response in SAMPLES:
        prompt = f"{problem}\n\nPlease reason step by step.\n\n"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids, step_spans = segment_response_token_spans(tokenizer, prompt_ids, response)
        input_ids = prompt_ids + response_ids
        response_span = (len(prompt_ids), len(input_ids))

        decoded = tokenizer.decode(input_ids[response_span[0] : response_span[1]])
        assert decoded.strip() == response.strip(), "step spans do not reconstruct the response"
        assert step_spans[0][0] == len(prompt_ids) and step_spans[-1][1] == len(input_ids)

        gradient_matrix = capture_sequence_gradients(
            model, torch.tensor([input_ids]), response_span, chunk_size=256
        )
        rows = [(start - response_span[0], end - response_span[0]) for start, end in step_spans]
        result = analyze_gradient_matrix(gradient_matrix, rows, energy_cutoff=0.95)

        selected = select_steps_by_energy(result.step_strengths, energy_threshold=0.95)
        stats = selection_stats(step_spans, selected)
        loss_mask = build_loss_mask(len(input_ids), step_spans, selected)

        assert gradient_matrix.shape == (response_span[1] - response_span[0], model.config.hidden_size)
        assert sum(loss_mask[: response_span[0]]) == 0, "prompt tokens must never be supervised"
        assert 0 < sum(loss_mask) <= response_span[1] - response_span[0]

        print(
            f"\n{problem[:45]}...\n"
            f"  tokens={len(input_ids)} steps={len(step_spans)} G={tuple(gradient_matrix.shape)} k*={result.k_star}\n"
            f"  strengths={[round(s, 4) for s in result.step_strengths]}\n"
            f"  selected steps={selected}  "
            f"dropped: steps={stats['step_drop']:.1%} tokens={stats['token_drop']:.1%}"
        )
        examples.append({"input_ids": input_ids, "loss_mask": loss_mask})

    collator = MaskedSFTCollator(pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    batch = collator(examples)
    labels = batch.pop("labels")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss = masked_cross_entropy(model(**batch).logits, labels)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    assert torch.isfinite(loss) and grad_norm > 0, "training step produced no usable gradient"
    print(f"\ntraining step: loss={loss.item():.4f} grad_norm={grad_norm:.4f}")
    print("PIPELINE SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
