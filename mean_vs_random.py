import argparse
import os
import clip
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from datasets import ImageNet
from utils import test_model_on_dataset
from merge import random_merge


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.join(os.environ.get("WORK", "."), "datasets"),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model-location",
        type=str,
        default=os.path.join(os.environ.get("WORK", "."), "models"),
        help="Where the models are stored.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Whether to overwrite existing results",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    RESULTS_FILE = "mean_vs_random_results.json"

    base_model, preprocess = clip.load("ViT-B/32", "cpu", jit=False)

    if os.path.exists(RESULTS_FILE) and not args.overwrite:
        with open(RESULTS_FILE, "r") as f:
            results = json.load(f)
        print(f"Loaded existing results from {RESULTS_FILE}")
    else:
        results = {"num_models": [], "uniform_accuracy": [], "random_accuracy": []}

    # Get sorted models based on ImageNet2p accuracy
    INDIVIDUAL_MODEL_RESULTS_FILE = "individual_model_results.jsonl"
    if not os.path.exists(INDIVIDUAL_MODEL_RESULTS_FILE):
        raise FileNotFoundError(
            f"Could not find {INDIVIDUAL_MODEL_RESULTS_FILE}. "
            "Please run main.py with --eval-individual-models first."
        )

    individual_model_db = pd.read_json(INDIVIDUAL_MODEL_RESULTS_FILE, lines=True)
    individual_model_val_accs = {}
    for _, row in individual_model_db.iterrows():
        individual_model_val_accs[row["model_name"]] = row["ImageNet2p"]

    sorted_models = sorted(
        individual_model_val_accs.items(), key=lambda x: x[1], reverse=True
    )
    sorted_model_names = [x[0] for x in sorted_models]

    model_paths = [
        os.path.join(args.model_location, f"{model_name}.pt")
        for model_name in sorted_model_names
    ]

    imagenet_dataset = ImageNet(
        preprocess, args.data_location, args.batch_size, args.workers
    )

    model_counts = [2, 5, 10, 15, 20, 30, 40, 50, 60, 70]

    for num_models in model_counts:
        if num_models in results["num_models"] and not args.overwrite:
            print(f"Skipping {num_models} models - already evaluated")
            continue

        print(f"\nEvaluating with top {num_models} models")
        selected_paths = model_paths[:num_models]

        uniform_model = random_merge(
            selected_paths,
            base_model,
            use_base=False,
            elect_sign=False,
            value="mean",
            cache=False,
        )
        uniform_accuracy = test_model_on_dataset(uniform_model, imagenet_dataset)
        print(f"Uniform merge accuracy: {uniform_accuracy:.2f}%")

        random_model = random_merge(
            selected_paths,
            base_model,
            use_base=False,
            elect_sign=False,
            value="random",
            cache=False,
        )
        random_accuracy = test_model_on_dataset(random_model, imagenet_dataset)
        print(f"Random merge accuracy: {random_accuracy:.2f}%")

        if num_models in results["num_models"]:
            idx = results["num_models"].index(num_models)
            results["uniform_accuracy"][idx] = uniform_accuracy
            results["random_accuracy"][idx] = random_accuracy
        else:
            results["num_models"].append(num_models)
            results["uniform_accuracy"].append(uniform_accuracy)
            results["random_accuracy"].append(random_accuracy)

        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        results["num_models"],
        results["uniform_accuracy"],
        "o-",
        label="Uniform Average",
    )
    plt.plot(
        results["num_models"], results["random_accuracy"], "s-", label="Random Merge"
    )
    plt.xlabel("Number of Models Merged", fontsize=14)
    plt.ylabel("ImageNet Accuracy (%)", fontsize=14)
    plt.title("Model Accuracy vs. Number of Merged Models", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("mean_vs_random.png")
    plt.close()

    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"Plot saved to mean_vs_random.png")


if __name__ == "__main__":
    main()
