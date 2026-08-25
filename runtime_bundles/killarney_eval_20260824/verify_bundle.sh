#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BUNDLE_DIR}"

sha256sum --check SHA256SUMS

if [[ "${VERIFY_ACTIVE_RUNTIME:-0}" == "1" ]]; then
  python verify_runtime.py
fi
