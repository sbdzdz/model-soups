import argparse
import os
import clip
import json
from pathlib import Path
from tabulate import tabulate
from tqdm import tqdm
import numpy as np
from typing import List
import matplotlib.pyplot as plt

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
from merge import merge


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
        default="quick_eval_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Whether to overwrite existing results",
    )
    parser.add_argument(
        "--merge-strategies",
        nargs="+",
        default=["random", "mean", "median"],
        help="Merge strategies to evaluate",
    )
    parser.add_argument(
        "--use-base",
        nargs="+",
        type=lambda x: x.lower() == "true",
        default=[False, True],
        help="Whether to use base model (True/False)",
    )
    parser.add_argument(
        "--elect-sign",
        nargs="+",
        type=lambda x: x.lower() == "true",
        default=[False, True],
        help="Whether to elect sign (True/False)",
    )
    parser.add_argument(
        "--alpha",
        nargs="+",
        type=float,
        default=[1.0],
        help="List of scaling factors for merged task vectors",
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
    for merge_strategy in args.merge_strategies:
        for use_base in args.use_base:
            for elect_sign in args.elect_sign:
                for variant in [
                    "all",
                    "transformer_only",
                    "attention_only",
                    "mlp_only",
                    "early_layers",
                    "late_layers",
                ]:
                    if use_base:
                        for alpha in args.alpha:
                            configs.append(
                                {
                                    "use_base": use_base,
                                    "elect_sign": elect_sign,
                                    "alpha": alpha,
                                    "value": merge_strategy,
                                    "variant": variant,
                                }
                            )
                    else:
                        configs.append(
                            {
                                "use_base": use_base,
                                "elect_sign": elect_sign,
                                "alpha": None,
                                "value": merge_strategy,
                                "variant": variant,
                            }
                        )

    for config in tqdm(configs, desc="Testing configurations"):
        config_name = config["value"]

        if config["elect_sign"]:
            config_name += "_elect_sign"

        if config["use_base"]:
            config_name += "_use_base"

        if config["alpha"] is not None:
            config_name += f"_alpha_{config['alpha']:.1f}"

        config_name += f"_{config['variant']}"

        if config_name not in results:
            results[config_name] = {"config": config}

        all_dataset_evaluated = all(
            dataset_name in results[config_name] for dataset_name in datasets_to_eval
        )
        if all_dataset_evaluated:
            print(f"Skipping {config_name} - all datasets already evaluated")
            continue

        print(f"\nMerging models with configuration: {config_name}")
        model = merge(
            model_paths,
            base_model,
            use_base=config["use_base"],
            elect_sign=config["elect_sign"],
            alpha=config["alpha"],
            value=config["value"],
            variant=config["variant"],
        )

        for dataset_name in datasets_to_eval:
            if dataset_name in results[config_name] and not args.overwrite:
                print(f"Skipping {dataset_name} for {config_name} - already evaluated")
                continue

            print(f"Evaluating on {dataset_name}...")
            dataset_cls = get_dataset_class(dataset_name)
            dataset = dataset_cls(
                preprocess, args.dataset_location, args.batch_size, args.workers
            )
            accuracy = test_model_on_dataset(model, dataset)
            results[config_name][dataset_name] = accuracy
            print(f"{dataset_name} accuracy: {accuracy * 100:.2f}%")

            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

    print("\n=== FINAL RESULTS ===")
    table_data = []
    headers = ["Configuration"] + datasets_to_eval

    for config_name, config_results in results.items():
        row = [config_name]
        for dataset in datasets_to_eval:
            if dataset in config_results:
                row.append(f"{config_results[dataset] * 100:.2f}%")
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

    # Create a scatter plot if datasets is set to "all"
    if "all" in args.datasets or set(datasets_to_eval) == set(
        [
            "imagenet2p",
            "imagenet",
            "imagenet_v2",
            "imagenet_sketch",
            "imagenet_r",
            "objectnet",
            "imagenet_a",
        ]
    ):
        create_scatter_plot(results)


def create_scatter_plot(results):

    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    value_colors = {
        "random": "C1",
        "mean": "C0",
        "median": "pink",
        "max": "purple",
    }
    markers = {False: "o", True: "^"}

    ood_datasets = [
        "imagenet_v2",
        "imagenet_sketch",
        "imagenet_r",
        "objectnet",
        "imagenet_a",
    ]

    for config_results in results.values():
        config = config_results["config"]
        value = config["value"]
        elect_sign = config["elect_sign"]
        use_base = config["use_base"]
        alpha = config["alpha"]
        variant = config["variant"]

        ood_accs = [
            config_results[dataset]
            for dataset in ood_datasets
            if dataset in config_results
        ]
        ood_acc = sum(ood_accs) / len(ood_accs) if ood_accs else 0

        imagenet_acc = config_results.get("imagenet", 0)

        label_parts = [value.capitalize()]
        if elect_sign:
            label_parts.append("elect sign")
        if use_base and alpha is not None:
            label_parts.append(f"alpha={alpha:.1f}")
        if variant:
            label_parts.append(variant)

        label = " ".join(
            [
                label_parts[0],
                f"({', '.join(label_parts[1:])})" if len(label_parts) > 1 else "",
            ]
        ).strip()

        if use_base:
            color_alpha = 0.4 + (0.5 * alpha) if alpha is not None else 0.65
            ax.scatter(
                imagenet_acc,
                ood_acc,
                marker=markers[elect_sign],
                color=value_colors.get(value, "gray"),
                s=200,
                label=label,
                alpha=color_alpha,
                zorder=5,
                linewidth=0,
            )
        else:
            ax.scatter(
                imagenet_acc,
                ood_acc,
                marker=markers[elect_sign],
                edgecolors=value_colors.get(value, "gray"),
                facecolors="none",
                s=200,
                label=label,
                linewidth=2,
                zorder=5,
            )

    ood_datasets = [
        "ImageNetV2",
        "ImageNetSketch",
        "ImageNetR",
        "ObjectNet",
        "ImageNetA",
    ]

    # Add greedy soup results
    with open("greedy_soup_results.jsonl", "r") as f:
        greedy_soup_data = json.loads(f.readline())

    ood_accs = [
        greedy_soup_data[dataset]
        for dataset in ood_datasets
        if dataset in greedy_soup_data
    ]
    greedy_ood_acc = sum(ood_accs) / len(ood_accs) if ood_accs else 0

    ax.scatter(
        greedy_soup_data.get("ImageNet", 0.0),
        greedy_ood_acc,
        marker="*",
        color="C4",
        s=400,
        label="Greedy Soup",
        zorder=10,
    )

    # Add base model results
    with open("individual_model_results.jsonl", "r") as f:
        base_model_data = json.loads(f.readline())

    base_ood_accs = [
        base_model_data[dataset]
        for dataset in ood_datasets
        if dataset in base_model_data
    ]
    base_ood_acc = sum(base_ood_accs) / len(base_ood_accs) if base_ood_accs else 0

    ax.scatter(
        base_model_data.get("ImageNet", 0.0),
        base_ood_acc,
        marker="h",
        color="slategray",
        s=150,
        label="Initialization (LP)",
        zorder=10,
    )

    # Set labels and grid
    ax.set_ylabel("Avg. accuracy on distribution shifts (%)", fontsize=16)
    ax.set_xlabel("ImageNet Accuracy (top-1%)", fontsize=16)
    ax.grid(True)

    # Add legend with unique entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(), by_label.keys(), fontsize=13, ncol=2, facecolor="white"
    ).set_zorder(100)

    plt.savefig("merge_comparison_plot.png", bbox_inches="tight")
    plt.close()
    print("Scatter plot saved as merge_comparison_plot.png")


if __name__ == "__main__":
    main()
