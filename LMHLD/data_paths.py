"""Resolve the existing dataset location without copying/downloading anything."""
import os
from pathlib import Path


def resolve_data_root(kind="native"):
    folder = "LMHLD_dataset_different_patch_sizes" if kind == "native" else "LMHLD_tiled"
    env_key = "KAT_DATA_ROOT" if kind == "native" else "KAT_TILED_ROOT"
    if os.environ.get(env_key):
        path = Path(os.environ[env_key])
        if not path.is_dir():
            raise FileNotFoundError(f"{env_key} directory is unavailable: {path}")
        return str(path)
    roots = [Path(__file__).resolve().parent / "datasets",
             Path(r"F:\LMHLD"), Path(r"F:\LMHLD数据集")]
    for root in roots:
        path = root / folder
        if path.is_dir():
            return str(path)
    # Preserve a readable legacy path for services that intentionally start without data.
    return str(roots[1] / folder)
