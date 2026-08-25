#!/usr/bin/env bash

_codex_restore_errexit=0
_codex_restore_nounset=0
_codex_restore_pipefail=0
case $- in
  *e*) _codex_restore_errexit=1 ;;
esac
case $- in
  *u*) _codex_restore_nounset=1 ;;
esac
if set -o | grep -q '^pipefail[[:space:]]*on$'; then
  _codex_restore_pipefail=1
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
module load python/3.11.5 arrow/23.0.1
if module is-loaded cuda/12.2 >/dev/null 2>&1; then
  module swap cuda/12.2 cuda/12.6
else
  module load cuda/12.6
fi
source "${VENV_OFFICIAL_ROOT}/bin/activate"

# Torch in vsi-official is built against CUDA 12.6. On this cluster the CUDA
# modules populate LIBRARY_PATH but not LD_LIBRARY_PATH, so add the runtime
# libraries explicitly for Python/torch dynamic loading.
if [[ -n "${EBROOTCUDA:-}" ]]; then
  _codex_cuda_runtime_dirs=(
    "${EBROOTCUDA}/targets/x86_64-linux/lib"
    "${EBROOTCUDA}/lib64"
    "${EBROOTCUDA}/lib"
    "${EBROOTCUDA}/nvvm/lib64"
    "${EBROOTCUDA}/extras/CUPTI/lib64"
  )
  for _codex_cuda_dir in "${_codex_cuda_runtime_dirs[@]}"; do
    if [[ -d "${_codex_cuda_dir}" ]] && [[ ":${LD_LIBRARY_PATH:-}:" != *":${_codex_cuda_dir}:"* ]]; then
      if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
        export LD_LIBRARY_PATH="${_codex_cuda_dir}:${LD_LIBRARY_PATH}"
      else
        export LD_LIBRARY_PATH="${_codex_cuda_dir}"
      fi
    fi
  done
  unset _codex_cuda_dir
  unset _codex_cuda_runtime_dirs
fi

if [[ -n "${PYTHONPATH:-}" ]]; then
  FILTERED_PYTHONPATH=""
  IFS=':' read -r -a _codex_pythonpath_parts <<< "${PYTHONPATH}"
  for _codex_part in "${_codex_pythonpath_parts[@]}"; do
    if [[ "${_codex_part}" == "/cvmfs/soft.computecanada.ca/easybuild/python/site-packages" ]]; then
      continue
    fi
    if [[ -z "${FILTERED_PYTHONPATH}" ]]; then
      FILTERED_PYTHONPATH="${_codex_part}"
    else
      FILTERED_PYTHONPATH="${FILTERED_PYTHONPATH}:${_codex_part}"
    fi
  done
  export PYTHONPATH="${FILTERED_PYTHONPATH}"
fi

if [[ -n "${ARROW_HOME:-}" ]]; then
  ARROW_PYTHONPATH="${ARROW_HOME}/lib/python3.11/site-packages"
  if [[ -z "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${ARROW_PYTHONPATH}"
  elif [[ ":${PYTHONPATH}:" != *":${ARROW_PYTHONPATH}:"* ]]; then
    export PYTHONPATH="${PYTHONPATH}:${ARROW_PYTHONPATH}"
  fi
fi

if [[ $_codex_restore_errexit -eq 1 ]]; then
  set -e
else
  set +e
fi

if [[ $_codex_restore_nounset -eq 1 ]]; then
  set -u
else
  set +u
fi

if [[ $_codex_restore_pipefail -eq 1 ]]; then
  set -o pipefail
else
  set +o pipefail
fi

unset _codex_restore_errexit
unset _codex_restore_nounset
unset _codex_restore_pipefail
