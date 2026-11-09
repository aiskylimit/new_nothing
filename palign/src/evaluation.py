import argparse
import json
import os
import signal

from math_verify import parse, verify
from tqdm import tqdm


def timeout(seconds=10):
    def decorator(func):
        def handler(signum, frame):
            raise TimeoutError("Verification timed out.")

        def wrapper(*args, **kwargs):
            if os.name != "posix":
                return func(*args, **kwargs)
            old = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)

        return wrapper

    return decorator


@timeout(seconds=10)
def label_with_math_verify(preds, golden):
    parsed_preds = list(map(parse, preds))
    parsed_golden = list(map(parse, ["$" + golden + "$"] * len(preds)))
    try:
        labels = list(map(verify, parsed_golden, parsed_preds))
    except Exception:
        labels = [0] * len(preds)
    return [int(x) for x in labels], parsed_preds


def evaluate_jsonl(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    results = []
    for item in tqdm(data, desc=input_file):
        outputs = item.get("output")
        if not outputs:
            continue
        if isinstance(outputs, str):
            outputs = [outputs]
        answer = str(item.get("answer", "")).strip()
        try:
            labels, parsed_outputs = label_with_math_verify(outputs, answer)
        except Exception:
            labels = [0] * len(outputs)
            parsed_outputs = [""] * len(outputs)
        item.update({"label": labels, "passn": int(any(labels)), "output_ans": parsed_outputs})
        results.append(item)
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", required=True)
    p.add_argument("--output_path", required=True)
    args = p.parse_args()
    evaluate_jsonl(args.input_path, args.output_path)
