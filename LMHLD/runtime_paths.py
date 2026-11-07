"""Keep this application's temporary files and model caches on the project drive."""
import os
import tempfile
from pathlib import Path


def configure_runtime_paths():
    root=Path(__file__).resolve().parent/'cache/runtime'
    if root.drive.upper()=='C:':raise RuntimeError('本项目缓存必须保存在D盘项目目录')
    mapping={'TEMP':'tmp','TMP':'tmp','TORCH_HOME':'torch','HF_HOME':'huggingface',
             'HF_HUB_CACHE':'huggingface/hub','XDG_CACHE_HOME':'packages',
             'MPLCONFIGDIR':'matplotlib','CUDA_CACHE_PATH':'cuda','NUMBA_CACHE_DIR':'numba',
             'PIP_CACHE_DIR':'pip','TIKTOKEN_CACHE_DIR':'tiktoken'}
    for name,relative in mapping.items():
        old=os.environ.get(name)
        if old and Path(old).drive.upper()=='D:':
            continue
        path=root/relative;path.mkdir(parents=True,exist_ok=True);os.environ[name]=str(path)
    tempfile.tempdir=os.environ['TEMP']
    return {name:os.environ.get(name) for name in mapping}
