import json
from pathlib import Path

from transformers import AutoTokenizer

from sgl.data.prepare import build_record
from sgl.selection import answer_steps
from sgl.training.dataset import MaskedSFTDataset

TOKENIZER = str(Path(__file__).parent / "regression" / "assets" / "tokenizer")


def test_only_answer_steps_and_the_stop_token_are_supervised(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    responses = [
        "First we add the numbers. That gives 42 here. Check the work again. The answer is \\boxed{42}.",
        "No final answer is written. We stop here.",
    ]
    records = []
    for index, response in enumerate(responses):
        records.append({**build_record(tokenizer, "q", response, max_tokens=4096), "id": index})
    data = tmp_path / "data.jsonl"
    data.write_text("".join(json.dumps(record) + "\n" for record in records))

    for definition, steps in (("mention", [1, 3]), ("boxed", [3])):
        output = tmp_path / f"{definition}.jsonl"
        answer_steps.main(["--data-path", str(data), "--tokenizer", TOKENIZER,
                           "--definition", definition, "--output-path", str(output)])
        (example,) = MaskedSFTDataset(str(output)).records  # the sample without an answer is dropped
        record = records[0]
        expected = {p for i in steps for p in range(record["steps"][i]["token_start"], record["steps"][i]["token_end"])}
        expected |= set(range(record["response_token_span"][1], len(record["input_ids"])))
        assert {p for p, bit in enumerate(example["loss_mask"]) if bit} == expected
