"""Import the untouched upstream code under references/ so tests can compare against it."""
import ast
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "references"
PALIGN = ROOT / "P-ALIGN" / "src"
SSFT = ROOT / "SegmentSelectiveSFT"


def load(path: Path, name: str, stubs: dict | None = None):
    """Import a file as module `name`; `stubs` pre-seeds sys.modules (e.g. vllm) for the import."""
    saved = {key: sys.modules.get(key) for key in stubs or {}}
    sys.modules.update(stubs or {})
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def functions_from(path: Path, names: list[str], namespace: dict) -> dict:
    """Exec only the named top-level functions of a script that does work at import time."""
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    missing = set(names) - {node.name for node in nodes}
    assert not missing, f"{path} lacks {missing}"
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def stub_module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module
