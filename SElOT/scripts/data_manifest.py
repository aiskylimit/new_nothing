#!/usr/bin/env python3
"""sha256 of every built data file (data/fit/*.jsonl, data/eval/*.jsonl), to tell whether a rebuild on another machine
gave the same data as the reference build.

    python scripts/data_manifest.py --write     # data_manifest.json from the current data/ (reference build)
    python scripts/data_manifest.py --check     # compare data/ with data_manifest.json; prints every difference

--check never fails the run (exit 0): a different build is reported, not refused. With --strict it exits 1.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data_manifest.json"


def digest(path: Path) -> dict:
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "lines": data.count(b"\n")}


def current() -> dict:
    return {str(p.relative_to(ROOT)): digest(p) for d in ("data/fit", "data/eval") for p in sorted((ROOT / d).glob("*.jsonl"))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    cur = current()
    if a.write:
        MANIFEST.write_text(json.dumps(cur, indent=1, sort_keys=True) + "\n")
        print(f"wrote {MANIFEST} ({len(cur)} files)")
        return
    ref = json.loads(MANIFEST.read_text())
    diff = []
    for k in sorted(set(ref) | set(cur)):
        if k not in cur:
            diff.append(f"missing    {k}")
        elif k not in ref:
            diff.append(f"extra      {k} ({cur[k]['lines']} lines)")
        elif cur[k]["sha256"] != ref[k]["sha256"]:
            diff.append(f"different  {k}: {cur[k]['lines']} lines (reference {ref[k]['lines']})")
    if diff:
        print(f"DATA DIFFERS from the reference build ({len(diff)} of {len(ref)} files):\n  " + "\n  ".join(diff))
        sys.exit(1 if a.strict else 0)
    print(f"data identical to the reference build ({len(ref)} files)")


if __name__ == "__main__":
    main()
