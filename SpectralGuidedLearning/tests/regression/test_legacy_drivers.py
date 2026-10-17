"""Every SGL config launches exactly the commands of the legacy driver that produced its published run."""
import shutil

import pytest
from legacy_drivers import run_all


@pytest.mark.regression
@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_configs_reproduce_legacy_driver_commands(tmp_path):
    problems = run_all(tmp_path)
    assert not problems, "\n".join(problems)
