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
from merge import merge

ALL_DATASETS = [
    "ImageNet2p",
    "ImageNet",
    "ImageNetV2",
    "ImageNetSketch",
    "ImageNetR",
    "ObjectNet",
    "ImageNetA",
]

OOD_DATASETS = [
    "ImageNetV2",
    "ImageNetSketch",
    "ImageNetR",
    "ObjectNet",
    "ImageNetA",
]


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
        default=["ImageNet2p"],
        choices=ALL_DATASETS + ["all"],
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
    return parser.parse_args()


def get_dataset_class(dataset_name):
    dataset_map = {
        "ImageNet2p": ImageNet2p,
        "ImageNet": ImageNet,
        "ImageNetV2": ImageNetV2,
        "ImageNetSketch": ImageNetSketch,
        "ImageNetR": ImageNetR,
        "ObjectNet": ObjectNet,
        "ImageNetA": ImageNetA,
    }
    return dataset_map.get(dataset_name)


def evaluate_modern_soup(alpha_model, datasets_to_eval, args, preprocess):
    """Evaluate the modern soup model on multiple datasets"""
    results = {}

    for dataset_name in datasets_to_eval:
        print(f"Evaluating modern soup on {dataset_name}...")
        dataset_cls = get_dataset_class(dataset_name)
        dataset = dataset_cls(
            preprocess, args.dataset_location, args.batch_size, args.workers
        )
        accuracy = test_model_on_dataset(alpha_model, dataset)
        results[dataset_name] = accuracy
        print(f"{dataset_name} accuracy: {accuracy * 100:.2f}%")

    # Calculate average OOD performance
    ood_accs = [results[dataset] for dataset in OOD_DATASETS if dataset in results]
    avg_ood_acc = sum(ood_accs) / len(ood_accs) if ood_accs else 0

    return results, avg_ood_acc


def add_to_scatter_plot(imagenet_acc, ood_acc):
    """Add modern soup results to existing scatter plot"""
    plt.figure()
    try:
        # Load existing plot if it exists
        img = plt.imread("merge_comparison_plot.png")
        plt.imshow(img)
    except:
        pass

    # Add our new point
    plt.scatter(
        imagenet_acc,
        ood_acc,
        marker="D",  # Diamond marker to distinguish from others
        color="red",
        s=400,
        label="Modern Learned Soup",
        zorder=15,  # Place it on top
    )

    plt.ylabel("Avg. accuracy on distribution shifts (%)", fontsize=16)
    plt.xlabel("ImageNet Accuracy (top-1%)", fontsize=16)
    plt.grid(True)

    # Update legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(
        by_label.values(), by_label.keys(), fontsize=13, ncol=2, facecolor="white"
    ).set_zorder(100)

    plt.savefig("merge_comparison_plot_with_modern.png", bbox_inches="tight")
    plt.close()
    print("Updated scatter plot saved as merge_comparison_plot_with_modern.png")


def main():
    args = parse_arguments()

    print("Loading base CLIP model...")
    base_model, preprocess = clip.load("ViT-B/32", "cpu", jit=False)

    model_dir = Path(args.model_location)
    model_paths = [str(model_dir / f"model_{i}.pt") for i in range(72)]

    datasets_to_eval = []
    if "all" in args.datasets:
        datasets_to_eval = ALL_DATASETS
    else:
        datasets_to_eval = args.datasets

    results = {}
    if os.path.exists(args.output) and not args.overwrite:
        with open(args.output, "r") as f:
            results = json.load(f)

    model = merge(model_paths, base_model)
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

    if "ImageNet2p" in datasets_to_eval:
        idx = datasets_to_eval.index("ImageNet2p") + 1
        table_data.sort(
            key=lambda x: float(x[idx].rstrip("%")) if x[idx] != "N/A" else 0,
            reverse=True,
        )

    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print(f"\nResults saved to {args.output}")

    if set(datasets_to_eval) == set(ALL_DATASETS):
        create_scatter_plot(results)

    # After training, evaluate on all specified datasets
    print("\nEvaluating final model on all datasets...")
    results, avg_ood_acc = evaluate_modern_soup(
        model, datasets_to_eval, args, preprocess
    )

    # Save results
    final_results = {
        "modern_learned_soup": {
            **results,
            "config": {
                "value": "learned",
                "elect_sign": False,
                "use_base": True,
                "alpha": None,
                "variant": "modern",
            },
        }
    }

    with open("modern_soup_results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    # Add to scatter plot if we have both ImageNet and OOD results
    if "ImageNet" in results:
        print("\nUpdating scatter plot with modern soup results...")
        add_to_scatter_plot(results["ImageNet"], avg_ood_acc)

    # Print final results table
    print("\n=== FINAL RESULTS ===")
    table_data = [["Dataset", "Accuracy"]]
    for dataset, acc in results.items():
        table_data.append([dataset, f"{acc * 100:.2f}%"])
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))


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

    for config_results in results.values():
        config = config_results["config"]
        value = config["value"]
        elect_sign = config["elect_sign"]
        use_base = config["use_base"]
        alpha = config["alpha"]
        variant = config["variant"]

        ood_accs = [
            config_results[dataset]
            for dataset in OOD_DATASETS
            if dataset in config_results
        ]
        ood_acc = sum(ood_accs) / len(ood_accs) if ood_accs else 0

        imagenet_acc = config_results.get("ImageNet", 0)

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

    # Add greedy soup results
    with open("greedy_soup_results.jsonl", "r") as f:
        greedy_soup_data = json.loads(f.readline())

    ood_accs = [
        greedy_soup_data[dataset]
        for dataset in OOD_DATASETS
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
        for dataset in OOD_DATASETS
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
