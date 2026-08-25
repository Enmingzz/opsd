#!/usr/bin/env bash

# Canonical runtime setup for the local Qwen3.6-27B strict evaluator.
# The vLLM extension and torch must come from the same isolated environment;
# inherited cluster Python/library paths can otherwise produce an ABI mismatch.

QWEN36_JUDGE_VENV=${QWEN36_JUDGE_VENV:-/scratch/enmingzz/temp/venvs/mmecot-qwen36-judge}
QWEN36_JUDGE_MODEL_PATH=${QWEN36_JUDGE_MODEL_PATH:-/scratch/enmingzz/models/Qwen3.6-27B-FP8}
QWEN36_JUDGE_SERVED_MODEL=${QWEN36_JUDGE_SERVED_MODEL:-Qwen/Qwen3.6-27B-FP8}
QWEN36_JUDGE_PYTHON=${QWEN36_JUDGE_PYTHON:-${QWEN36_JUDGE_VENV}/bin/python}
QWEN36_JUDGE_VLLM=${QWEN36_JUDGE_VLLM:-${QWEN36_JUDGE_VENV}/bin/vllm}

qwen36_prepare_runtime() {
  local site_packages=${QWEN36_JUDGE_VENV}/lib/python3.11/site-packages

  [[ -x "${QWEN36_JUDGE_PYTHON}" ]]
  [[ -x "${QWEN36_JUDGE_VLLM}" ]]
  [[ -f "${QWEN36_JUDGE_MODEL_PATH}/config.json" ]]
  [[ -d "${site_packages}/torch/lib" ]]
  [[ -d "${site_packages}/nvidia/cu13/lib" ]]

  unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX
  export PYTHONNOUSERSITE=1
  export TOKENIZERS_PARALLELISM=false
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export PYTHONUNBUFFERED=1
  export LD_LIBRARY_PATH="${site_packages}/torch/lib:${site_packages}/nvidia/cu13/lib:/usr/lib64/nvidia"

  "${QWEN36_JUDGE_PYTHON}" - <<'PY'
import torch
import vllm
import vllm._C_stable_libtorch  # noqa: F401

assert torch.__version__ == "2.11.0+cu130", torch.__version__
assert vllm.__version__ == "0.25.1", vllm.__version__
print(f"qwen36_runtime_ok torch={torch.__version__} vllm={vllm.__version__}")
PY
}

export QWEN36_JUDGE_VENV QWEN36_JUDGE_MODEL_PATH QWEN36_JUDGE_SERVED_MODEL
export QWEN36_JUDGE_PYTHON QWEN36_JUDGE_VLLM
