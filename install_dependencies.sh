#!/usr/bin/env bash
set -euo pipefail

USE_SYSTEM=0
if [[ "${1:-}" == "--system" ]]; then
  USE_SYSTEM=1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Installing/upgrading uv..."
"${PYTHON_BIN}" -m pip install --upgrade --quiet uv

UV_PIP_ARGS=()
if [[ "${USE_SYSTEM}" == "1" ]]; then
  UV_PIP_ARGS+=(--system)
else
  if [[ ! -d ".venv" ]]; then
    echo "Creating local virtual environment at .venv..."
    uv venv --python "${PYTHON_BIN}" .venv
  fi
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "Warning: this dependency stack is mainly for Linux/CUDA."
  echo "Some packages such as unsloth, xformers, bitsandbytes, causal_conv1d, or flash-linear-attention may fail on macOS."
fi

read -r NUMPY_SPEC PILLOW_SPEC < <("${PYTHON_BIN}" - <<'PY'
try:
    import numpy
    numpy_spec = f"numpy=={numpy.__version__}"
except Exception:
    numpy_spec = "numpy"

try:
    import PIL
    pillow_spec = f"pillow=={PIL.__version__}"
except Exception:
    pillow_spec = "pillow"

print(numpy_spec, pillow_spec)
PY
)

TORCH_OR_COLAB_MISSING="$("${PYTHON_BIN}" - <<'PY'
import importlib.util
import os

torch_missing = importlib.util.find_spec("torch") is None
is_colab = "COLAB_" in "".join(os.environ.keys())
print("1" if torch_missing or is_colab else "0")
PY
)"

UNSLOTH_MISSING="$("${PYTHON_BIN}" - <<'PY'
import importlib.util

print("1" if importlib.util.find_spec("unsloth") is None else "0")
PY
)"

if [[ "${TORCH_OR_COLAB_MISSING}" == "1" ]]; then
  echo "Installing torch/CUDA and unsloth stack..."
  uv pip install "${UV_PIP_ARGS[@]}" \
    "torch==2.8.0" "triton>=3.3.0" "${NUMPY_SPEC}" "${PILLOW_SPEC}" \
    torchvision bitsandbytes xformers==0.0.32.post2 \
    "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
    "unsloth[base] @ git+https://github.com/unslothai/unsloth"

  uv pip install "${UV_PIP_ARGS[@]}" --no-deps "torchcodec==0.7.0"
elif [[ "${UNSLOTH_MISSING}" == "1" ]]; then
  echo "Installing unsloth..."
  uv pip install "${UV_PIP_ARGS[@]}" unsloth
fi

echo "Installing remaining training dependencies..."
uv pip install "${UV_PIP_ARGS[@]}" --upgrade --no-deps \
  "tokenizers>=0.22.0,<=0.23.0" trl==0.22.2 unsloth unsloth_zoo

uv pip install "${UV_PIP_ARGS[@]}" transformers==5.2.0

uv pip install "${UV_PIP_ARGS[@]}" --no-build-isolation \
  flash-linear-attention causal_conv1d==1.6.0

uv pip install "${UV_PIP_ARGS[@]}" --no-deps --upgrade "torchao>=0.16.0"

uv pip install "${UV_PIP_ARGS[@]}" gdown

echo "Done."
if [[ "${USE_SYSTEM}" != "1" ]]; then
  echo "Activate the environment with: source .venv/bin/activate"
fi
