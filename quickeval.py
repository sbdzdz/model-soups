import argparse
import os
import clip
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils import ModelWrapper

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
from specops import WeightedMergeLayer, WeightedMergeModel, WeightedMergeSpectrum

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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_file = Path("results/quickeval.jsonl")
    if output_file.exists():
        with open(output_file) as f:
            results = [json.loads(line) for line in f]

    existing_variants = [result["model_name"] for result in results]
    if all(variant in existing_variants for variant in args.variants):
        print("No evaluation needed. Creating plot directly.")
        create_comparison_plot(output_file)
        return

    print("Loading base CLIP model...")
    base_model, preprocess = clip.load("ViT-B/32", device, jit=False)

    model_dir = args.model_location
    model_paths = [model_dir / f"model_{i}.pt" for i in range(72)]

    print("Loading model checkpoints...")
    raw_checkpoints = [torch.load(path, map_location="cpu") for path in model_paths]
    checkpoints = [
        {
            k: v.requires_grad_(False) if isinstance(v, torch.Tensor) else v
            for k, v in checkpoint.items()
        }
        for checkpoint in raw_checkpoints
    ]

    datasets_to_eval = []
    if args.datasets == ["all"] or "all" in args.datasets:
        datasets_to_eval = ALL_DATASETS
    else:
        for dataset in args.datasets:
            assert dataset in ALL_DATASETS, f"Dataset {dataset} not in {ALL_DATASETS}"
            datasets_to_eval.append(dataset)

    feature_dim = checkpoints[0]["classification_head.weight"].shape[1]
    num_classes = checkpoints[0]["classification_head.weight"].shape[0]
    model = ModelWrapper(base_model, feature_dim, num_classes, normalize=True)

    for alpha_path, variant in zip(args.alphas_paths, args.variants):
        existing_result = next((r for r in results if r["model_name"] == variant), None)
        if existing_result and not args.overwrite:
            print(f"Skipping evaluation for {variant}.")
            continue

        print(f"Loading alpha values from {alpha_path}...")
        alpha_values = torch.load(alpha_path, map_location=device)

        if variant == "model":
            weighted_model = WeightedMergeModel(
                model,
                checkpoints,
                normalize=False,
            )
            weighted_model._alpha = alpha_values
        elif variant == "layer":
            weighted_model = WeightedMergeLayer(
                model,
                checkpoints,
                normalize=False,
            )
            weighted_model._alpha = alpha_values
        elif variant == "spectrum":
            weighted_model = WeightedMergeSpectrum(
                model,
                checkpoints,
                num_singular_values=alpha_values.shape[1],
                normalize=False,
            )
            weighted_model.alpha = alpha_values
        else:
            raise ValueError(f"Unknown weighting scheme: {variant}")

        weighted_model = weighted_model.to(device)

        result = {"model_name": variant}
        for dataset_name in datasets_to_eval:
            print(f"Evaluating {variant} on {dataset_name}...")
            dataset_cls = get_dataset_class(dataset_name)
            dataset = dataset_cls(
                preprocess, args.dataset_location, args.batch_size, args.workers
            )
            accuracy = test_model_on_dataset(weighted_model, dataset)
            result[dataset_name] = accuracy
            print(f"{dataset_name} accuracy: {accuracy * 100:.2f}%")

        if existing_result:
            results[results.index(existing_result)] = result
        else:
            results.append(result)

        with open(output_file, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Results saved to {output_file}")

    create_comparison_plot(output_file)


def create_comparison_plot(output_file):
    """
    Create a scatter plot comparing all models using results from jsonl files,
    matching the style from main.py.

    Args:
        specops_results_file: Path to the quickeval results jsonl file
    """
    plt.figure(figsize=(12, 12))
    ax = plt.gca()

    path = Path("results/individual_model.jsonl")
    individual_models = []
    with open(path, "r") as f:
        for line in f:
            individual_models.append(json.loads(line))

    print(f"Loaded {len(individual_models)} individual models")

    for model in individual_models:
        ood_accs = [model[dataset] for dataset in OOD_DATASETS]
        model["OOD"] = sum(ood_accs) / len(ood_accs)

    base_model = individual_models[0]
    ax.scatter(
        base_model["ImageNet"],
        base_model["OOD"],
        marker="h",
        color="slategray",
        s=150,
        label="Initialization (LP)",
        zorder=10,
    )
    print(
        f"Plotted Initialization: ImageNet={base_model['ImageNet']:.4f}, OOD={base_model['OOD']:.4f}"
    )

    other_models = individual_models[1:]
    imagenet_accs = [model["ImageNet"] for model in other_models]
    ood_accs = [model["OOD"] for model in other_models]

    ax.scatter(
        imagenet_accs,
        ood_accs,
        marker="d",
        color="C2",
        s=130,
        label="Various checkpoints",
        zorder=9,
    )
    print(f"Plotted {len(other_models)} additional individual checkpoints")

    result_files = {
        "greedy_soup": Path("results/greedy_soup.jsonl"),
        "uniform_soup": Path("results/uniform_soup.jsonl"),
    }

    for path in result_files.values():
        assert path.exists(), f"Results file {path} does not exist"

    results_data = {}

    with open(output_file, "r") as f:
        results = [json.loads(line) for line in f]
        print(f"Loaded {len(results)} results")

    for model_type, file_path in result_files.items():
        with open(file_path, "r") as f:
            results_data[model_type] = json.loads(f.readline())
            print(f"Loaded {model_type} results from {file_path}")

        data = results_data[model_type]
        ood_accs = [data[dataset] for dataset in OOD_DATASETS]
        data["OOD"] = sum(ood_accs) / len(ood_accs)

    colors = plt.cm.Set2.colors[: len(results)]
    for i, entry in enumerate(results):
        ood_accs = [entry[dataset] for dataset in OOD_DATASETS]
        entry["OOD"] = sum(ood_accs) / len(ood_accs)

        variant = entry["model_name"]
        color = colors[i]

        ax.scatter(
            entry["ImageNet"],
            entry["OOD"],
            marker="o",
            color=color,
            s=100,
            label=f"Learned merge ({variant})",
            zorder=100,
        )
        print(
            f"Plotted {model_type}: ImageNet={entry['ImageNet']:.4f}, OOD={entry['OOD']:.4f}"
        )

    uniform_data = results_data["uniform_soup"]
    ax.scatter(
        uniform_data["ImageNet"],
        uniform_data["OOD"],
        marker="o",
        color="C0",
        s=100,
        label="Uniform Soup",
        zorder=10,
    )
    print(
        f"Plotted Uniform Soup: ImageNet={uniform_data['ImageNet']:.4f}, OOD={uniform_data['OOD']:.4f}"
    )

    greedy_data = results_data["greedy_soup"]
    ax.scatter(
        greedy_data["ImageNet"],
        greedy_data["OOD"],
        marker="o",
        color="C4",
        s=100,
        label="Greedy Soup",
        zorder=10,
    )
    print(
        f"Plotted Greedy Soup: ImageNet={greedy_data['ImageNet']:.4f}, OOD={greedy_data['OOD']:.4f}"
    )

    ax.set_ylabel("Avg. accuracy on distribution shifts (%)", fontsize=16)
    ax.set_xlabel("ImageNet Accuracy (top-1%)", fontsize=16)
    ax.grid(True)

    ax.set_ylim(0.36, 0.52)
    ax.set_xlim(0.745, 0.82)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        fontsize=13,
        ncol=2,
        facecolor="white",
        loc="lower right",
    ).set_zorder(100)

    plt.savefig("merge_comparison_plot.png", bbox_inches="tight")
    plt.close()
    print("Scatter plot saved as merge_comparison_plot.png")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate specops model and create comparison plot"
    )
    parser.add_argument(
        "--model-location",
        type=Path,
        default=Path(os.environ.get("WORK", ".")) / "models",
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--dataset-location",
        type=Path,
        default=Path(os.environ.get("WORK", ".")) / "datasets",
        help="Root directory for datasets",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Datasets to evaluate on. Use 'all' for all datasets or specify individual datasets by name.",
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
        "--alphas-paths",
        nargs="+",
        type=Path,
        required=True,
        help="List of paths to the saved alpha weights files (.pt)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["spectrum", "model", "layer"],
        required=True,
        help="List of weighting schemes used for model merging. Must match number of alphas-paths.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results files",
    )
    args = parser.parse_args()

    if len(args.alphas_paths) != len(args.variants):
        raise ValueError("Number of alphas-paths must match number of variants")

    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
