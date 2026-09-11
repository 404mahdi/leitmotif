"""Leitmotif: graph neural networks + BERT for understanding musical context."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Keep model downloads and library caches inside the repo instead of the user
# profile. This runs on the first `import src...`, before transformers or
# torch.hub read these variables.
_CACHE = ROOT / ".cache"
for _var, _folder in {
    "HF_HOME": "huggingface",
    "TORCH_HOME": "torch",
    "MPLCONFIGDIR": "matplotlib",
    "NUMBA_CACHE_DIR": "numba",
}.items():
    os.environ.setdefault(_var, str(_CACHE / _folder))
