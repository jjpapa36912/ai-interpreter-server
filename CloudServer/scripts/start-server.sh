#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from pathlib import Path
import os

adapter = Path(os.environ["AI_INTERPRETER_TRANSLATION_ADAPTER"])
required = adapter / "adapter_model.safetensors"
if not required.is_file() or required.stat().st_size < 100_000_000:
    raise SystemExit(f"invalid translation adapter: {required}")
PY

# ctranslate2 currently links against CUDA 12 cublas while the NGC runtime may
# expose CUDA 13. The wheel below supplies the ABI it needs without modifying
# the host driver or requiring the manual symlink procedure used in prototypes.
CUBLAS_DIR="$(python -c 'import importlib.util,pathlib; spec=importlib.util.find_spec("nvidia.cublas"); print(pathlib.Path(next(iter(spec.submodule_search_locations))) / "lib")')"
export LD_LIBRARY_PATH="${CUBLAS_DIR}:${LD_LIBRARY_PATH:-}"

exec python -m uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1
