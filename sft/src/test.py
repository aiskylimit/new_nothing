"""Compatibility entrypoint for the shared P-ALIGN vLLM generator."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[2] / "palign/src/test.py"), run_name="__main__")
