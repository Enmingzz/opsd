#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <train|eval> /path/to/clean/VLMEvalKit" >&2
  exit 2
fi

mode=$1
checkout=$(realpath "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
base_commit=51682a6baab948d3dbb4b867a3eab178504ac3f5

case "$mode" in
  train)
    patch_file="$script_dir/train_checkout.patch"
    hash_file="$script_dir/train_files.sha256"
    ;;
  eval)
    patch_file="$script_dir/eval_checkout.patch"
    hash_file="$script_dir/eval_files.sha256"
    ;;
  *)
    echo "Mode must be 'train' or 'eval', got: $mode" >&2
    exit 2
    ;;
esac

git -C "$checkout" rev-parse --is-inside-work-tree >/dev/null
actual_commit=$(git -C "$checkout" rev-parse HEAD)
if [[ "$actual_commit" != "$base_commit" ]]; then
  echo "Expected VLMEvalKit $base_commit, found $actual_commit" >&2
  exit 1
fi

expected_paths=$(awk '{print $2}' "$hash_file" | sort)
actual_paths=$(git -C "$checkout" diff --name-only | sort)
if (cd "$checkout" && sha256sum --quiet -c "$hash_file" >/dev/null 2>&1) \
    && [[ "$actual_paths" == "$expected_paths" ]]; then
  echo "$mode checkout already matches the captured Killarney tree."
  exit 0
fi

if [[ -n "$(git -C "$checkout" status --porcelain)" ]]; then
  echo "Refusing to patch a dirty checkout that does not exactly match the target." >&2
  git -C "$checkout" status --short >&2
  exit 1
fi

(cd "$script_dir" && sha256sum --quiet -c patches.sha256)
git -C "$checkout" apply --check "$patch_file"
git -C "$checkout" apply "$patch_file"
git -C "$checkout" diff --check

actual_paths=$(git -C "$checkout" diff --name-only | sort)
if [[ "$actual_paths" != "$expected_paths" ]]; then
  echo "Patched file set does not match the manifest." >&2
  exit 1
fi
(cd "$checkout" && sha256sum -c "$hash_file")
echo "$mode checkout restored and verified at $checkout"
