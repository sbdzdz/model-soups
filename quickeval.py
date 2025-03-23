import argparse
import os
import clip
import json
from pathlib import Path
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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate specops model and create comparison plot"
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
        "--alphas-path",
        type=str,
        required=True,
        help="Path to the saved alpha weights file (.pt)",
    )
    parser.add_argument(
        "--weighting",
        type=str,
        choices=["spectrum", "model", "layer"],
        required=True,
        help="Weighting scheme used for model merging",
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


def main():
    args = parse_arguments()

    print("Loading base CLIP model...")
    base_model, preprocess = clip.load("ViT-B/32", "cpu", jit=False)

    model_dir = Path(args.model_location)
    model_paths = [str(model_dir / f"model_{i}.pt") for i in range(72)]

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
        # Verify that all provided dataset names are valid
        for dataset in args.datasets:
            if dataset not in ALL_DATASETS:
                print(
                    f"Warning: Unknown dataset '{dataset}'. Available datasets: {', '.join(ALL_DATASETS)}"
                )
            else:
                datasets_to_eval.append(dataset)

        if not datasets_to_eval:
            print("No valid datasets specified, using all datasets.")
            datasets_to_eval = ALL_DATASETS

    if args.alphas_path and args.weighting:
        from utils import ModelWrapper

        feature_dim = checkpoints[0]["classification_head.weight"].shape[1]
        num_classes = checkpoints[0]["classification_head.weight"].shape[0]
        model = ModelWrapper(base_model, feature_dim, num_classes, normalize=True)

        print(f"Loading alpha values from {args.alphas_path}...")
        alpha_values = torch.load(args.alphas_path, map_location="cpu")

        if args.weighting == "layer":
            weighted_model = WeightedMergeLayer(model, checkpoints, unnormalised=False)
            weighted_model._alpha = torch.nn.Parameter(alpha_values)
        elif args.weighting == "spectrum":
            weighted_model = WeightedMergeSpectrum(
                model, checkpoints, num_singular_values=alpha_values.shape[1]
            )
            weighted_model.alpha = torch.nn.Parameter(alpha_values)
        else:
            weighted_model = WeightedMergeModel(model, checkpoints, unnormalised=False)
            weighted_model._alpha = torch.nn.Parameter(alpha_values)

        model_name = f"specops_{args.weighting}"

        results = {"model_name": model_name}
        for dataset_name in datasets_to_eval:
            print(f"Evaluating on {dataset_name}...")
            dataset_cls = get_dataset_class(dataset_name)
            dataset = dataset_cls(
                preprocess, args.dataset_location, args.batch_size, args.workers
            )
            accuracy = test_model_on_dataset(weighted_model, dataset)
            results[dataset_name] = accuracy
            print(f"{dataset_name} accuracy: {accuracy * 100:.2f}%")

        output_file = f"specops_{args.weighting}_results.jsonl"
        with open(output_file, "w") as f:
            f.write(json.dumps(results) + "\n")
        print(f"Results saved to {output_file}")
        create_comparison_plot(output_file)


def create_comparison_plot(specops_results_file):
    """
    Create a scatter plot comparing all models using results from jsonl files.

    Args:
        specops_results_file: Path to the specops results jsonl file
    """
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Define colors and markers for different model types
    model_styles = {
        "specops_model": {
            "color": "red",
            "marker": "D",
            "size": 300,
            "label": "SpecOps (model)",
        },
        "specops_layer": {
            "color": "green",
            "marker": "D",
            "size": 300,
            "label": "SpecOps (layer)",
        },
        "specops_spectrum": {
            "color": "blue",
            "marker": "D",
            "size": 300,
            "label": "SpecOps (spectrum)",
        },
        "greedy_soup": {
            "color": "C4",
            "marker": "*",
            "size": 400,
            "label": "Greedy Soup",
        },
        "uniform_soup": {
            "color": "orange",
            "marker": "P",
            "size": 300,
            "label": "Uniform Soup",
        },
        "individual_models": {
            "color": "slategray",
            "marker": "h",
            "size": 150,
            "label": "Initialization (LP)",
        },
    }

    result_files = [
        specops_results_file,
        "greedy_soup_results.jsonl",
        "uniform_soup_results.jsonl",
        "individual_model_results.jsonl",
    ]

    for file_path in result_files:
        try:
            with open(file_path, "r") as f:
                results = json.loads(f.readline())

            model_name = results.get("model_name", Path(file_path).stem)

            ood_accs = [
                results[dataset] for dataset in OOD_DATASETS if dataset in results
            ]
            ood_acc = sum(ood_accs) / len(ood_accs) if ood_accs else 0
            imagenet_acc = results.get("ImageNet", 0)

            style = next(
                (v for k, v in model_styles.items() if k in model_name),
                {"color": "gray", "marker": "o", "size": 200, "label": model_name},
            )

            ax.scatter(
                imagenet_acc,
                ood_acc,
                marker=style["marker"],
                color=style["color"],
                s=style["size"],
                label=style["label"],
                zorder=10,
            )

            print(
                f"Plotted {model_name}: ImageNet={imagenet_acc:.4f}, Avg OOD={ood_acc:.4f}"
            )

        except FileNotFoundError:
            print(f"Warning: Could not load {file_path}")

    ax.set_ylabel("Avg. accuracy on distribution shifts (%)", fontsize=16)
    ax.set_xlabel("ImageNet Accuracy (top-1%)", fontsize=16)
    ax.grid(True)

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
