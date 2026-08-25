#!/usr/bin/env bash

ENV_SCRIPT="/project/6101803/enmingzz/vlm/scripts/activate_official_env.sh"

if [[ ! -f "${ENV_SCRIPT}" ]]; then
  echo "Missing activation script: ${ENV_SCRIPT}" >&2
  return 1 2>/dev/null || exit 1
fi

source "${ENV_SCRIPT}"
