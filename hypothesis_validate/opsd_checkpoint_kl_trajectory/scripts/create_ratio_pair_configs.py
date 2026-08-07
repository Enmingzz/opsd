#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = "0.15:0.175,0.15:0.20,0.175:0.20,0.20:0.225"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create OPSD/SFT configs for budget-gap ratio pairs.")
    parser.add_argument("--pairs", default=DEFAULT_PAIRS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "configs" / "ratio_pairs")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ratio_tag(value: float) -> str:
    percentage = 100.0 * float(value)
    if abs(percentage - round(percentage)) < 1e-9:
        return f"r{round(percentage):03d}"
    tenths_of_percent = round(1000.0 * float(value))
    return f"r{tenths_of_percent:04d}"


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    source_configs = {
        "opsd": ROOT / "config_max1024.json",
        "sft": ROOT / "config_sft_max1024.json",
    }
    output_base = ROOT / "outputs" / "ratio_pairs_mmstar_clean100_max1024"
    pairs: list[tuple[float, float]] = []
    for raw_pair in args.pairs.split(","):
        low_raw, high_raw = raw_pair.strip().split(":", maxsplit=1)
        low, high = float(low_raw), float(high_raw)
        if not 0.0 < low < high <= 1.0:
            raise ValueError(f"Invalid ratio pair: {raw_pair}")
        pairs.append((low, high))

    manifest: list[dict[str, Any]] = []
    for low, high in pairs:
        pair_tag = f"{ratio_tag(low)}_{ratio_tag(high)}"
        for method, source_path in source_configs.items():
            cfg = json.loads(source_path.read_text(encoding="utf-8"))
            cfg["experiment_id"] = (
                f"llm_only_{method}_checkpoint_kl_mmstar_clean100_"
                f"{pair_tag}_max1024_20260729"
            )
            cfg["rollout_retention_ratio"] = low
            cfg["comparison_retention_ratio"] = high
            cfg["output_root"] = str(output_base / pair_tag / method)
            target = output_dir / f"config_{method}_{pair_tag}_max1024.json"
            if target.exists() and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite {target}; pass --overwrite")
            atomic_write_json(target, cfg)
            manifest.append(
                {
                    "pair": pair_tag,
                    "low_retention_ratio": low,
                    "high_retention_ratio": high,
                    "method": method,
                    "config": str(target),
                    "checkpoint_root": cfg["checkpoint_root"],
                    "output_root": cfg["output_root"],
                    "checkpoint_count": len(cfg["checkpoint_steps"]),
                }
            )

    atomic_write_json(output_dir / "manifest.json", manifest)
    print(f"created_configs={len(manifest)}")
    print(f"manifest={output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
