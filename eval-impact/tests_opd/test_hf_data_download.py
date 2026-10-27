import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from opd.local_assets import download, prepare_reference_dolly, verify_dolly_snapshot, parse_manifest


def snapshot_fixture(target):
    target = Path(target)
    (target / "original").mkdir(parents=True, exist_ok=True)
    files = []
    for split in ("train", "dev", "valid"):
        path = target / "original" / f"{split}.jsonl"
        path.write_text(json.dumps({"prompt": split, "output": "reference"}) + "\n")
        files.append({"path": f"original/{split}.jsonl",
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (target / "manifest.json").write_text(json.dumps({"files": files}))


class HFDataDownloadTests(unittest.TestCase):
    def test_default_manifest_keeps_template_format_and_pins_dataset_urls(self):
        project = Path(__file__).resolve().parents[1]
        reference = json.loads((project / "HF_DATASET.json").read_text())
        with tempfile.TemporaryDirectory() as temp:
            entries = parse_manifest(project / "download.txt", temp)
        self.assertTrue(all(e["kind"] in ("--hf", "--url") for e in entries))
        dataset = [e for e in entries if e["kind"] == "--url"]
        self.assertEqual(len(dataset), 4)
        for entry in dataset:
            self.assertIn(f"/datasets/{reference['repo_id']}/resolve/{reference['revision']}/", entry["source"])

    def test_dataset_download_uses_pinned_revision_and_dataset_repo_type(self):
        reference = json.loads((Path(__file__).resolve().parents[1] / "HF_DATASET.json").read_text())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "download.txt"
            manifest.write_text("--hf-dataset pkcii/eval-impact-dolly /mnt/local/test/impact_opd/raw/dolly_reference\n")
            with patch("huggingface_hub.snapshot_download", side_effect=lambda **kw: snapshot_fixture(kw["local_dir"])) as call:
                download(manifest, root)
            self.assertEqual(call.call_args.kwargs["repo_type"], "dataset")
            self.assertEqual(call.call_args.kwargs["revision"], reference["revision"])
            self.assertIn("original/*.jsonl", call.call_args.kwargs["allow_patterns"])
            saved = json.loads((root / "download_manifest.json").read_text())["entries"][0]
            self.assertEqual(saved["resolved_commit"], reference["revision"])
            self.assertTrue((root / "raw/dolly_reference/original/train.jsonl").is_file())

    def test_corrupt_downloaded_original_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            snapshot_fixture(temp)
            verify_dolly_snapshot(temp)
            (Path(temp) / "original/train.jsonl").write_text("tampered\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_dolly_snapshot(temp)

    def test_prepare_requires_downloaded_local_data(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(FileNotFoundError, "Download Dolly first"):
                prepare_reference_dolly(temp)


if __name__ == "__main__":
    unittest.main()
