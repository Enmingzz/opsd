#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from vlmeval.config import supported_VLM
from vlmeval.dataset import build_dataset
from vlmeval.inference import infer_data
from vlmeval.smp import dump, get_pred_file_path, load


def shard_path(pred_root: Path, rank: int, world_size: int, dataset_name: str) -> Path:
    return pred_root / f"{rank}{world_size}_{dataset_name}.pkl"


def fill_missing_shard(
    pred_root: Path,
    model_name: str,
    dataset_name: str,
    rank: int,
    world_size: int,
) -> None:
    out_file = shard_path(pred_root, rank, world_size, dataset_name)
    if out_file.exists():
        print(f"[repair] shard exists, skip infer: {out_file}", flush=True)
        return

    dataset = build_dataset(dataset_name)
    if dataset is None:
        raise RuntimeError(f"Failed to build dataset {dataset_name}")

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    model = supported_VLM[model_name]()
    infer_data(
        model=model,
        model_name=model_name,
        work_dir=str(pred_root),
        dataset=dataset,
        out_file=str(out_file),
        verbose=False,
        api_nproc=4,
        use_vllm=False,
    )
    print(f"[repair] wrote shard: {out_file}", flush=True)


def merge_and_evaluate(
    pred_root: Path,
    model_name: str,
    dataset_name: str,
    world_size: int,
) -> Path:
    dataset = build_dataset(dataset_name)
    if dataset is None:
        raise RuntimeError(f"Failed to build dataset {dataset_name}")

    data_all = {}
    missing = []
    for rank in range(world_size):
        path = shard_path(pred_root, rank, world_size, dataset_name)
        if not path.exists():
            missing.append(str(path))
            continue
        data_all.update(load(str(path)))
    if missing:
        raise FileNotFoundError("Missing shard files:\n" + "\n".join(missing))

    data = dataset.data.copy()
    missing_indices = [x for x in data["index"] if x not in data_all]
    if missing_indices:
        raise RuntimeError(f"{dataset_name} missing {len(missing_indices)} predictions")

    vals = [data_all[x] for x in data["index"]]
    if all(isinstance(v, dict) and "prediction" in v and "extra_records" in v for v in vals):
        data["prediction"] = [v["prediction"] for v in vals]
        data["extra_records"] = [v["extra_records"] for v in vals]
    else:
        data["prediction"] = [str(v) for v in vals]
    if "image" in data:
        data.pop("image")

    result_file = Path(get_pred_file_path(str(pred_root), model_name, dataset_name, use_env_format=True))
    dump(data, str(result_file))
    print(f"[repair] merged {dataset_name}: {result_file}", flush=True)

    eval_results = dataset.evaluate(
        str(result_file),
        nproc=4,
        verbose=False,
        retry=3,
        model="chatgpt-0125",
    )
    print(f"[repair] evaluated {dataset_name}: {eval_results}", flush=True)
    return result_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--model-name", default="Qwen")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--fill-pope-rank", type=int, default=1)
    parser.add_argument("--skip-fill-pope", action="store_true")
    args = parser.parse_args()

    pred_root = Path(args.pred_root)
    if not pred_root.is_dir():
        raise FileNotFoundError(pred_root)

    if not args.skip_fill_pope:
        fill_missing_shard(
            pred_root=pred_root,
            model_name=args.model_name,
            dataset_name="POPE",
            rank=args.fill_pope_rank,
            world_size=args.world_size,
        )

    for dataset_name in ["MMStar", "POPE"]:
        merge_and_evaluate(
            pred_root=pred_root,
            model_name=args.model_name,
            dataset_name=dataset_name,
            world_size=args.world_size,
        )


if __name__ == "__main__":
    main()
