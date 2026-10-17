"""Every CPU pipeline stage must reproduce the pre-refactor outputs byte for byte (see pipeline.py)."""
import json

import pytest
from pipeline import GOLDEN, diff, run_pipeline


@pytest.mark.regression
def test_pipeline_outputs_match_golden(tmp_path):
    problems = diff(json.loads(GOLDEN.read_text()), run_pipeline(tmp_path))
    assert not problems, "\n".join(problems)
