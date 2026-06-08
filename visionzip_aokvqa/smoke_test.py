#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


OUTPUT_ROOT = Path("outputs/visionzip_aokvqa_reasoning")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--allow_embedding_fallback", action="store_true")
    p.add_argument("--log_path", default=str(OUTPUT_ROOT / "logs" / "smoke_test.log"))
    return p


def run(cmd: list[str], log_file) -> None:
    log_file.write("$ " + " ".join(cmd) + "\n")
    log_file.flush()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_file.write(proc.stdout)
    log_file.write(f"\n[exit_code] {proc.returncode}\n")
    log_file.flush()
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    methods = ["sft", "grpo", "epic", "opsd"]
    config_paths = {method: f"configs/experiments/aokvqa_visionzip_{method}.yaml" for method in methods}
    with log_path.open("w", encoding="utf-8") as log_file:
        for method in methods:
            cmd = [
                sys.executable,
                "-m",
                "opsd.visionzip_aokvqa.train",
                "--config",
                config_paths[method],
                "--method",
                method,
                "--smoke",
                "--output_dir",
                str(OUTPUT_ROOT / "checkpoints" / method / "smoke"),
            ]
            if args.allow_embedding_fallback:
                cmd.append("--allow_embedding_fallback")
            run(cmd, log_file)
            eval_cmd = [
                sys.executable,
                "-m",
                "opsd.visionzip_aokvqa.evaluate",
                "--config",
                config_paths[method],
                "--method",
                method,
                "--checkpoint_path",
                str(OUTPUT_ROOT / "checkpoints" / method / "smoke" / "final"),
                "--benchmark",
                "AOKVQA-smoke",
                "--retention_ratio",
                "0.1",
                "--limit",
                "1",
                "--output_jsonl",
                str(OUTPUT_ROOT / "eval" / method / "smoke_r0.1.jsonl"),
            ]
            if args.allow_embedding_fallback:
                eval_cmd.append("--allow_embedding_fallback")
            run(eval_cmd, log_file)
    print(f"smoke log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
