import argparse
import json
import os
import signal

from math_verify import parse, verify
from tqdm import tqdm


def timeout(seconds=10):
    def decorator(function):
        def handler(signum, frame):
            raise TimeoutError("verification timed out")

        def wrapper(*args, **kwargs):
            if os.name != "posix":
                return function(*args, **kwargs)
            previous = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                return function(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous)

        return wrapper

    return decorator


@timeout(seconds=10)
def label_with_math_verify(predictions, golden):
    parsed_predictions = list(map(parse, predictions))
    parsed_golden = list(map(parse, ["$" + golden + "$"] * len(predictions)))
    try:
        labels = list(map(verify, parsed_golden, parsed_predictions))
    except Exception:  # noqa: BLE001 - verifier raises heterogeneous parser errors
        labels = [0] * len(predictions)
    return [int(value) for value in labels], parsed_predictions


def evaluate_jsonl(input_file, output_file, expected_n=3):
    with open(input_file, "r", encoding="utf-8") as handle:
        data = [json.loads(line) for line in handle if line.strip()]
    results = []
    for index, item in enumerate(tqdm(data, desc=input_file)):
        outputs = item.get("output")
        if isinstance(outputs, str):
            outputs = [outputs]
        if not isinstance(outputs, list) or len(outputs) != expected_n:
            actual = (
                len(outputs) if isinstance(outputs, list) else type(outputs).__name__
            )
            raise RuntimeError(
                f"{input_file}: row {index} expected {expected_n} outputs, found {actual}"
            )
        if any(not isinstance(output, str) for output in outputs):
            raise TypeError(f"{input_file}: row {index} contains a non-string output")
        answer = str(item.get("answer", "")).strip()
        try:
            labels, parsed_outputs = label_with_math_verify(outputs, answer)
        except Exception:  # noqa: BLE001 - one malformed answer must not abort a benchmark
            labels = [0] * len(outputs)
            parsed_outputs = [""] * len(outputs)
        item.update(
            {"label": labels, "passn": int(any(labels)), "output_ans": parsed_outputs}
        )
        results.append(item)
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        handle.writelines(
            json.dumps(result, ensure_ascii=False, default=str) + "\n"
            for result in results
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--expected_n", type=int, default=3)
    arguments = parser.parse_args()
    if arguments.expected_n <= 0:
        parser.error("--expected_n must be positive")
    evaluate_jsonl(arguments.input_path, arguments.output_path, arguments.expected_n)
