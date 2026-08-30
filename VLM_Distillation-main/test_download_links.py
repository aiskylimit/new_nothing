#!/home/vinhld8/miniconda3/envs/dagger/bin/python
"""Check a download.txt without downloading full datasets or model weights.

URL entries download only a small byte range to a temporary directory.  Hugging
Face entries download only config.json with the `hf` executable from the dagger
environment.  The destination paths in download.txt are parsed and reported,
but are never created or modified.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


DAGGER_PYTHON = Path("/home/vinhld8/miniconda3/envs/dagger/bin/python")
DAGGER_HF = Path("/home/vinhld8/miniconda3/envs/dagger/bin/hf")
CURL = Path("/usr/bin/curl")


@dataclass(frozen=True)
class Entry:
    line_number: int
    kind: str
    source: str
    destination: str


def parse_download_file(path: Path) -> tuple[list[Entry], list[str]]:
    entries: list[Entry] = []
    errors: list[str] = []

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            fields = shlex.split(line, comments=False, posix=True)
        except ValueError as exc:
            errors.append(f"line {line_number}: cannot parse: {exc}")
            continue
        if len(fields) != 3:
            errors.append(
                f"line {line_number}: expected 3 fields TAG SOURCE DEST, got {len(fields)}"
            )
            continue
        kind, source, destination = fields
        if kind not in {"--url", "--hf"}:
            errors.append(f"line {line_number}: unsupported tag {kind!r}")
            continue
        if kind == "--url" and not source.startswith(("http://", "https://")):
            errors.append(f"line {line_number}: invalid HTTP(S) URL {source!r}")
            continue
        if kind == "--hf" and (source.count("/") != 1 or source.startswith("/")):
            errors.append(f"line {line_number}: invalid Hugging Face repo id {source!r}")
            continue
        entries.append(Entry(line_number, kind, source, destination))

    return entries, errors


def run(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )


def test_url(entry: Entry, temp_dir: Path, byte_count: int, timeout: int) -> tuple[bool, str]:
    output = temp_dir / f"url-line-{entry.line_number}.part"
    command = [
        str(CURL),
        "--location",
        "--fail",
        "--silent",
        "--show-error",
        "--connect-timeout",
        str(min(timeout, 15)),
        "--max-time",
        str(timeout),
        "--range",
        f"0-{byte_count - 1}",
        "--max-filesize",
        str(max(byte_count * 4, 65536)),
        "--output",
        str(output),
        "--write-out",
        "%{http_code}\t%{size_download}\t%{url_effective}",
        entry.source,
    ]
    try:
        result = run(command, timeout + 5)
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"

    status = result.stdout.strip()
    # curl code 63 means the server advertised/returned more than max-filesize.
    # A successful HTTP status still proves that the link is reachable, while
    # preventing an accidental full download when Range is ignored.
    status_fields = status.split("\t", 2)
    http_code = status_fields[0] if status_fields else "000"
    downloaded = status_fields[1] if len(status_fields) > 1 else "?"
    final_host = urlsplit(status_fields[2]).hostname if len(status_fields) > 2 else None
    summary = f"HTTP {http_code}; {downloaded} bytes"
    if final_host:
        summary += f"; final host={final_host}"
    reachable = http_code.startswith("2") or http_code == "304"
    if result.returncode == 0 or (result.returncode == 63 and reachable):
        note = "reachable"
        if result.returncode == 63:
            note += "; server ignored Range or file exceeds safety limit"
        return True, f"{note}; {summary}"
    detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "curl failed"
    return False, f"{summary}; {detail} (curl rc={result.returncode})"


def test_hf(entry: Entry, temp_dir: Path, timeout: int) -> tuple[bool, str]:
    local_dir = temp_dir / f"hf-line-{entry.line_number}"
    command = [
        str(DAGGER_HF),
        "download",
        entry.source,
        "config.json",
        "--local-dir",
        str(local_dir),
        "--quiet",
    ]
    try:
        result = run(command, timeout)
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s while downloading config.json"
    if result.returncode == 0 and (local_dir / "config.json").is_file():
        size = (local_dir / "config.json").stat().st_size
        return True, f"downloaded config.json ({size} bytes)"
    detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "hf failed"
    return False, f"{detail} (hf rc={result.returncode})"


def select(entries: list[Entry], kind: str, limit: int) -> list[Entry]:
    selected = [entry for entry in entries if entry.kind == kind]
    return selected if limit == 0 else selected[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "download_file",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("download.txt"),
        help="download.txt to check (default: the file next to this script)",
    )
    parser.add_argument("--url-limit", type=int, default=3, help="URL rows to test; 0 means all")
    parser.add_argument("--hf-limit", type=int, default=2, help="HF rows to test; 0 means all")
    parser.add_argument("--bytes", type=int, default=4096, help="maximum requested bytes per URL")
    parser.add_argument("--timeout", type=int, default=60, help="timeout in seconds per check")
    args = parser.parse_args()

    if Path(sys.executable).resolve() != DAGGER_PYTHON.resolve():
        print(f"ERROR: run with {DAGGER_PYTHON}, not {sys.executable}", file=sys.stderr)
        return 2
    if not args.download_file.is_file():
        print(f"ERROR: file not found: {args.download_file}", file=sys.stderr)
        return 2
    if args.url_limit < 0 or args.hf_limit < 0 or args.bytes < 1 or args.timeout < 1:
        print("ERROR: limits must be >= 0; bytes and timeout must be >= 1", file=sys.stderr)
        return 2
    if not CURL.is_file() or not DAGGER_HF.is_file():
        print(f"ERROR: required existing command missing: {CURL} or {DAGGER_HF}", file=sys.stderr)
        return 2

    entries, parse_errors = parse_download_file(args.download_file)
    print(f"Parsed {len(entries)} entries from {args.download_file}")
    for error in parse_errors:
        print(f"[SYNTAX FAIL] {error}")
    if parse_errors:
        return 1

    chosen = select(entries, "--url", args.url_limit) + select(entries, "--hf", args.hf_limit)
    if not chosen:
        print("No entries selected.")
        return 0

    passed = 0
    with tempfile.TemporaryDirectory(prefix="download-link-test-") as temp:
        temp_dir = Path(temp)
        for entry in chosen:
            print(f"\nline {entry.line_number}: {entry.kind} {entry.source}")
            print(f"  declared destination (not written): {entry.destination}")
            if entry.kind == "--url":
                ok, detail = test_url(entry, temp_dir, args.bytes, args.timeout)
            else:
                ok, detail = test_hf(entry, temp_dir, args.timeout)
            print(f"  [{'PASS' if ok else 'FAIL'}] {detail}")
            passed += int(ok)

    print(f"\nSummary: {passed}/{len(chosen)} checks passed")
    return 0 if passed == len(chosen) else 1


if __name__ == "__main__":
    raise SystemExit(main())
