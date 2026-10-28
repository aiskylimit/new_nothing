#!/usr/bin/env python3
"""Register project adapters and execute the existing VLMEvalKit runner."""

import os
import runpy
import sys
from pathlib import Path


def main():
    project = Path(os.environ["PROJECT_DIR"]).resolve()
    kit = Path(os.environ["VLMEVALKIT_DIR"]).resolve()
    context = Path(os.environ["EVAL_CONTEXT"]).resolve()
    sys.path[:0] = [str(project), str(kit), str(project / "scripts/eval")]
    import vlmeval.dataset
    import vlmeval.smp
    import vlmeval.vlm
    from eval_utils import update_summary
    from vlmeval_datasets import (LimitedImageMCQDataset, LimitedImageVQADataset,
                                  MMEPerceptionDataset)
    from vlmeval_models import (FastVLMChat, OfflineScoringModel,
                                ProjectQwen2VLChat, ProjectQwen3VLChat)
    vlmeval.vlm.FastVLMChat = FastVLMChat
    vlmeval.vlm.ProjectQwen2VLChat = ProjectQwen2VLChat
    vlmeval.vlm.ProjectQwen3VLChat = ProjectQwen3VLChat
    vlmeval.vlm.OfflineScoringModel = OfflineScoringModel
    vlmeval.dataset.MMEPerceptionDataset = MMEPerceptionDataset
    vlmeval.dataset.LimitedImageMCQDataset = LimitedImageMCQDataset
    vlmeval.dataset.LimitedImageVQADataset = LimitedImageVQADataset
    original = vlmeval.smp.upsert_dataset_status

    def mirrored_status(*args, **kwargs):
        result = original(*args, **kwargs)
        try:
            update_summary(context)
        except Exception as exc:
            print(f"WARNING: canonical summary update failed: {exc}", file=sys.stderr)
        return result

    vlmeval.smp.upsert_dataset_status = mirrored_status
    sys.argv = [str(kit / "run.py"), *sys.argv[1:]]
    try:
        runpy.run_path(str(kit / "run.py"), run_name="__main__")
    finally:
        update_summary(context)


if __name__ == "__main__":
    main()
