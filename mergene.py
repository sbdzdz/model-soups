import argparse
import os
import clip
import json
from pathlib import Path
from tabulate import tabulate
from tqdm import tqdm
import numpy as np
from typing import List, Optional
import torch

from datasets import (
    ImageNet2p,
    ImageNet,
    ImageNetV2,
    ImageNetSketch,
    ImageNetR,
    ObjectNet,
    ImageNetA,
)
from utils import test_model_on_dataset
from merge import random_merge


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Merge models and evaluate performance"
    )
    parser.add_argument(
        "--model-location",
        type=str,
        default=os.path.join(os.environ.get("WORK", "."), "models"),
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--dataset-location",
        type=str,
        default=os.path.join(os.environ.get("WORK", "."), "datasets"),
        help="Root directory for datasets",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["imagenet2p"],
        choices=[
            "imagenet2p",
            "imagenet",
            "imagenet_v2",
            "imagenet_sketch",
            "imagenet_r",
            "objectnet",
            "imagenet_a",
            "all",
        ],
        help="Datasets to evaluate on",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker threads for data loading",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="merge_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Whether to overwrite existing results",
    )
    return parser.parse_args()


def get_dataset_class(dataset_name):
    dataset_map = {
        "imagenet2p": ImageNet2p,
        "imagenet": ImageNet,
        "imagenet_v2": ImageNetV2,
        "imagenet_sketch": ImageNetSketch,
        "imagenet_r": ImageNetR,
        "objectnet": ObjectNet,
        "imagenet_a": ImageNetA,
    }
    return dataset_map.get(dataset_name)


def main():
    args = parse_arguments()

    print("Loading base CLIP model...")
    base_model, preprocess = clip.load("ViT-B/32", "cpu", jit=False)

    model_dir = Path(args.model_location)
    model_paths = [str(model_dir / f"model_{i}.pt") for i in range(72)]

    datasets_to_eval = []
    if "all" in args.datasets:
        datasets_to_eval = [
            "imagenet2p",
            "imagenet",
            "imagenet_v2",
            "imagenet_sketch",
            "imagenet_r",
            "objectnet",
            "imagenet_a",
        ]
    else:
        datasets_to_eval = args.datasets

    results = {}
    if os.path.exists(args.output) and not args.overwrite:
        with open(args.output, "r") as f:
            results = json.load(f)

    configs = []
    for merge_strategy in ["random", "mean", "median"]:
        for use_base in [False, True]:
            for elect_sign in [False, True]:
                if use_base:
                    for alpha in np.arange(0.1, 1.3, 0.1):
                        configs.append(
                            {
                                "use_base": use_base,
                                "elect_sign": elect_sign,
                                "alpha": alpha,
                                "value": merge_strategy,
                            }
                        )
                else:
                    configs.append(
                        {
                            "use_base": use_base,
                            "elect_sign": elect_sign,
                            "alpha": None,
                            "value": merge_strategy,
                        }
                    )

    for config in tqdm(configs, desc="Testing configurations"):
        config_name = config["value"]

        if config["elect_sign"]:
            config_name += "_elect_sign"

        if config["use_base"]:
            config_name += "_use_base"

        if config["alpha"] is not None:
            config_name += f"alpha_{config['alpha']:.1f}"

        if config_name in results and not args.overwrite:
            print(f"Skipping {config_name} - already evaluated")
            continue

        print(f"\nMerging models with configuration: {config_name}")
        model = random_merge(
            model_paths,
            base_model,
            use_base=config["use_base"],
            elect_sign=config["elect_sign"],
            alpha=config["alpha"],
            value=config["value"],
        )

        config_results = {"config": config}
        for dataset_name in datasets_to_eval:
            dataset_cls = get_dataset_class(dataset_name)
            if dataset_cls:
                print(f"Evaluating on {dataset_name}...")
                dataset = dataset_cls(
                    preprocess, args.dataset_location, args.batch_size, args.workers
                )
                accuracy = test_model_on_dataset(model, dataset)
                config_results[dataset_name] = accuracy
                print(f"{dataset_name} accuracy: {accuracy:.2f}%")
            else:
                print(f"Unknown dataset: {dataset_name}")

        results[config_name] = config_results

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    print("\n=== FINAL RESULTS ===")
    table_data = []
    headers = ["Configuration"] + datasets_to_eval

    for config_name, config_results in results.items():
        row = [config_name]
        for dataset in datasets_to_eval:
            if dataset in config_results:
                row.append(f"{config_results[dataset]:.2f}%")
            else:
                row.append("N/A")
        table_data.append(row)

    if "imagenet2p" in datasets_to_eval:
        idx = datasets_to_eval.index("imagenet2p") + 1
        table_data.sort(
            key=lambda x: float(x[idx].rstrip("%")) if x[idx] != "N/A" else 0,
            reverse=True,
        )

    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
