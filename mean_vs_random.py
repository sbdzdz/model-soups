import argparse
import os
import clip
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from datasets import ImageNet
from utils import test_model_on_dataset
from merge import merge


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
        default=1024,
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

        if "uniform_accuracy" not in results:
            results["uniform_accuracy"] = [None] * len(results["num_models"])

        if "random_accuracy" not in results:
            results["random_accuracy"] = [None] * len(results["num_models"])

        if "random_elect_sign_accuracy" not in results:
            results["random_elect_sign_accuracy"] = [None] * len(results["num_models"])

    else:
        results = {
            "num_models": [],
            "uniform_accuracy": [],
            "random_accuracy": [],
            "random_elect_sign_accuracy": [],
        }

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
        # First check if this number of models is already in our results
        if num_models in results.get("num_models", []):
            idx = results["num_models"].index(num_models)

            # Check if each merge type needs evaluation
            need_uniform = (
                args.overwrite
                or "uniform_accuracy" not in results
                or results["uniform_accuracy"][idx] is None
            )
            need_random = (
                args.overwrite
                or "random_accuracy" not in results
                or results["random_accuracy"][idx] is None
            )
            need_random_elect = (
                args.overwrite
                or "random_elect_sign_accuracy" not in results
                or results["random_elect_sign_accuracy"][idx] is None
            )
        else:
            # If this number of models isn't in our results yet, we need to evaluate everything
            need_uniform = True
            need_random = True
            need_random_elect = True

            # Add this number of models to our results
            results.setdefault("num_models", []).append(num_models)
            results.setdefault("uniform_accuracy", []).append(None)
            results.setdefault("random_accuracy", []).append(None)
            results.setdefault("random_elect_sign_accuracy", []).append(None)
            idx = len(results["num_models"]) - 1

        if not (need_uniform or need_random or need_random_elect):
            print(f"Skipping {num_models} models - already evaluated all merge types")
            continue

        print(f"\nEvaluating with top {num_models} models")
        selected_paths = model_paths[:num_models]

        if need_uniform:
            uniform_model = merge(
                selected_paths,
                base_model,
                use_base=False,
                elect_sign=False,
                value="mean",
                cache=False,
            )
            uniform_accuracy = test_model_on_dataset(uniform_model, imagenet_dataset)
            print(f"Uniform merge accuracy: {uniform_accuracy:.2f}%")
            results["uniform_accuracy"][idx] = uniform_accuracy

        if need_random:
            random_model = merge(
                selected_paths,
                base_model,
                use_base=False,
                elect_sign=False,
                value="random",
                cache=False,
            )
            random_accuracy = test_model_on_dataset(random_model, imagenet_dataset)
            print(f"Random merge accuracy: {random_accuracy:.2f}%")
            results["random_accuracy"][idx] = random_accuracy

        if need_random_elect:
            random_elect_model = merge(
                selected_paths,
                base_model,
                use_base=False,
                elect_sign=True,
                value="random",
                cache=False,
            )
            random_elect_accuracy = test_model_on_dataset(
                random_elect_model, imagenet_dataset
            )
            print(f"Random merge (elect sign) accuracy: {random_elect_accuracy:.2f}%")
            results["random_elect_sign_accuracy"][idx] = random_elect_accuracy

        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        results["num_models"],
        results["uniform_accuracy"],
        "o-",
        color="tab:blue",
        label="Uniform",
    )
    plt.plot(
        results["num_models"],
        results["random_accuracy"],
        "s-",
        color="tab:red",
        label="Random",
    )

    plt.plot(
        results["num_models"],
        results["random_elect_sign_accuracy"],
        "^-",
        color="tab:green",
        label="Random (elect sign)",
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
