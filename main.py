import argparse
import os
import wget
import torch
import clip
import os
from pathlib import Path
import json
import operator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets import (
    ImageNet2p,
    ImageNet,
    ImageNetV2,
    ImageNetSketch,
    ImageNetR,
    ObjectNet,
    ImageNetA,
)
from utils import get_model_from_sd, test_model_on_dataset
from merge import merge, get_cache_path


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser("~/data"),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model-location",
        type=str,
        default=os.path.expanduser("~/ssd/checkpoints/soups"),
        help="Where to download the models.",
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--eval-individual-models",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--uniform-soup",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--greedy-soup",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--randmerge",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="all",
        choices=["all", "best"],
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
        "--alpha",
        type=float,
        nargs="+",
        default=[1.0],
        help="List of scaling factors for merged task vectors",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Whether to overwrite existing results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    NUM_MODELS = 72
    INDIVIDUAL_MODEL_RESULTS_FILE = Path("results/individual_model.jsonl")
    UNIFORM_SOUP_RESULTS_FILE = Path("results/uniform_soup.jsonl")
    GREEDY_SOUP_RESULTS_FILE = Path("results/greedy_soup.jsonl")
    RANDOM_MERGE_RESULTS_FILE = Path("results/random_merge.jsonl")

    # Step 1: Download models.
    if args.download_models:
        if not os.path.exists(args.model_location):
            os.mkdir(args.model_location)
        for i in range(NUM_MODELS):
            model_path = os.path.join(args.model_location, f"model_{i}.pt")
            if os.path.exists(model_path):
                print(f"\nSkipping model {i} - file already exists")
            else:
                print(f"\nDownloading model {i} of {NUM_MODELS - 1}")
                wget.download(
                    f"https://github.com/mlfoundations/model-soups/releases/download/v0.0.2/model_{i}.pt",
                    out=args.model_location,
                )

    model_paths = [
        os.path.join(args.model_location, f"model_{i}.pt") for i in range(NUM_MODELS)
    ]

    # Step 2: Evaluate individual models.
    if (
        args.eval_individual_models
        or args.uniform_soup
        or args.greedy_soup
        or args.randmerge
    ):
        base_model, preprocess = clip.load("ViT-B/32", "cpu", jit=False)

    if args.eval_individual_models:
        if os.path.exists(INDIVIDUAL_MODEL_RESULTS_FILE):
            os.remove(INDIVIDUAL_MODEL_RESULTS_FILE)
        for j, model_path in enumerate(model_paths):
            assert os.path.exists(model_path)
            state_dict = torch.load(model_path, map_location=torch.device("cpu"))
            model = get_model_from_sd(state_dict, base_model)

            results = {"model_name": f"model_{j}"}
            # Note: ImageNet2p is the held-out minival set from ImageNet train that we use.
            # It is called 2p for 2 percent of ImageNet, or 26k images.
            # See utils on how this dataset is handled slightly differently.
            for dataset_cls in [
                ImageNet2p,
                ImageNet,
                ImageNetV2,
                ImageNetSketch,
                ImageNetR,
                ObjectNet,
                ImageNetA,
            ]:

                print(
                    f"Evaluating model {j} of {NUM_MODELS - 1} on {dataset_cls.__name__}."
                )

                dataset = dataset_cls(
                    preprocess, args.data_location, args.batch_size, args.workers
                )
                accuracy = test_model_on_dataset(model, dataset)
                results[dataset_cls.__name__] = accuracy
                print(accuracy)

            with open(INDIVIDUAL_MODEL_RESULTS_FILE, "a+") as f:
                f.write(json.dumps(results) + "\n")

    # Step 3: Uniform Soup.
    if args.uniform_soup:
        if os.path.exists(UNIFORM_SOUP_RESULTS_FILE):
            os.remove(UNIFORM_SOUP_RESULTS_FILE)

        # create the uniform soup sequentially to not overload memory
        for j, model_path in enumerate(model_paths):

            print(f"Adding model {j} of {NUM_MODELS - 1} to uniform soup.")

            assert os.path.exists(model_path)
            state_dict = torch.load(model_path)
            if j == 0:
                uniform_soup = {
                    k: v * (1.0 / NUM_MODELS) for k, v in state_dict.items()
                }
            else:
                uniform_soup = {
                    k: v * (1.0 / NUM_MODELS) + uniform_soup[k]
                    for k, v in state_dict.items()
                }

        model = get_model_from_sd(uniform_soup, base_model)

        results = {"model_name": f"uniform_soup"}
        for dataset_cls in [
            ImageNet2p,
            ImageNet,
            ImageNetV2,
            ImageNetSketch,
            ImageNetR,
            ObjectNet,
            ImageNetA,
        ]:

            print(f"Evaluating on {dataset_cls.__name__}.")

            dataset = dataset_cls(
                preprocess, args.data_location, args.batch_size, args.workers
            )
            accuracy = test_model_on_dataset(model, dataset)
            results[dataset_cls.__name__] = accuracy
            print(accuracy)

        with open(UNIFORM_SOUP_RESULTS_FILE, "a+") as f:
            f.write(json.dumps(results) + "\n")

    # Step 4: Greedy Soup.
    if args.greedy_soup:
        if os.path.exists(GREEDY_SOUP_RESULTS_FILE):
            os.remove(GREEDY_SOUP_RESULTS_FILE)

        # Sort models by decreasing accuracy on the held-out validation set ImageNet2p
        # (We call the held out-val set ImageNet2p because it is 2 percent of ImageNet train)
        individual_model_db = pd.read_json(INDIVIDUAL_MODEL_RESULTS_FILE, lines=True)
        individual_model_val_accs = {}
        for _, row in individual_model_db.iterrows():
            individual_model_val_accs[row["model_name"]] = row["ImageNet2p"]
        individual_model_val_accs = sorted(
            individual_model_val_accs.items(), key=operator.itemgetter(1)
        )
        individual_model_val_accs.reverse()
        sorted_models = [x[0] for x in individual_model_val_accs]

        # Start the soup by using the first ingredient.
        greedy_soup_ingredients = [sorted_models[0]]
        greedy_soup_params = torch.load(
            os.path.join(args.model_location, f"{sorted_models[0]}.pt")
        )
        best_val_acc_so_far = individual_model_val_accs[0][1]
        held_out_val_set = ImageNet2p(
            preprocess, args.data_location, args.batch_size, args.workers
        )

        # Now, iterate through all models and consider adding them to the greedy soup.
        for i in range(1, NUM_MODELS):
            print(f"Testing model {i} of {NUM_MODELS}")

            # Get the potential greedy soup, which consists of the greedy soup with the new model added.
            new_ingredient_params = torch.load(
                os.path.join(args.model_location, f"{sorted_models[i]}.pt")
            )
            num_ingredients = len(greedy_soup_ingredients)
            potential_greedy_soup_params = {
                k: greedy_soup_params[k].clone()
                * (num_ingredients / (num_ingredients + 1.0))
                + new_ingredient_params[k].clone() * (1.0 / (num_ingredients + 1))
                for k in new_ingredient_params
            }

            # Run the potential greedy soup on the held-out val set.
            model = get_model_from_sd(potential_greedy_soup_params, base_model)
            held_out_val_accuracy = test_model_on_dataset(model, held_out_val_set)

            # If accuracy on the held-out val set increases, add the new model to the greedy soup.
            print(
                f"Potential greedy soup val acc {held_out_val_accuracy}, best so far {best_val_acc_so_far}."
            )
            if held_out_val_accuracy > best_val_acc_so_far:
                greedy_soup_ingredients.append(sorted_models[i])
                best_val_acc_so_far = held_out_val_accuracy
                greedy_soup_params = potential_greedy_soup_params
                print(f"Adding to soup. New soup is {greedy_soup_ingredients}")

        # Finally, evaluate the greedy soup.
        model = get_model_from_sd(greedy_soup_params, base_model)
        results = {"model_name": f"greedy_soup"}
        for dataset_cls in [
            ImageNet2p,
            ImageNet,
            ImageNetV2,
            ImageNetSketch,
            ImageNetR,
            ObjectNet,
            ImageNetA,
        ]:
            print(f"Evaluating on {dataset_cls.__name__}.")
            dataset = dataset_cls(
                preprocess, args.data_location, args.batch_size, args.workers
            )
            accuracy = test_model_on_dataset(model, dataset)
            results[dataset_cls.__name__] = accuracy
            print(accuracy)

        with open(GREEDY_SOUP_RESULTS_FILE, "a+") as f:
            f.write(json.dumps(results) + "\n")

    # Step 5: Random Merge
    if args.randmerge:
        if args.overwrite and os.path.exists(RANDOM_MERGE_RESULTS_FILE):
            os.remove(RANDOM_MERGE_RESULTS_FILE)

        existing_configs = set()
        if os.path.exists(RANDOM_MERGE_RESULTS_FILE):
            with open(RANDOM_MERGE_RESULTS_FILE, "r") as f:
                for line in f:
                    result = json.loads(line)
                    existing_configs.add(result["model_name"])

        use_base_options = [True, False]
        elect_sign_options = [True, False]
        value_options = ["random", "mean", "median", "max"]

        for use_base in use_base_options:
            for elect_sign in elect_sign_options:
                for value in value_options:
                    alphas = args.alpha if use_base else [None]

                    for alpha in alphas:
                        config_name = (
                            get_cache_path(value, elect_sign, use_base, alpha)
                            .split("/")[-1]
                            .replace(".pt", "")
                        )

                        if config_name in existing_configs and not args.overwrite:
                            print(f"\nSkipping {config_name} - already exists")
                            continue

                        model = merge(
                            model_paths,
                            base_model,
                            use_base=use_base,
                            elect_sign=elect_sign,
                            value=value,
                            alpha=alpha,
                        )

                        results = {"model_name": config_name}
                        results.update(
                            {
                                "params": {
                                    "value": value,
                                    "elect_sign": elect_sign,
                                    "use_base": use_base,
                                    "alpha": alpha,
                                }
                            }
                        )

                        for dataset_cls in [
                            ImageNet2p,
                            ImageNet,
                            ImageNetV2,
                            ImageNetSketch,
                            ImageNetR,
                            ObjectNet,
                            ImageNetA,
                        ]:
                            print(
                                f"Evaluating {config_name} on {dataset_cls.__name__}."
                            )
                            dataset = dataset_cls(
                                preprocess,
                                args.data_location,
                                args.batch_size,
                                args.workers,
                            )
                            accuracy = test_model_on_dataset(model, dataset)
                            results[dataset_cls.__name__] = accuracy
                            print(accuracy)

                        with open(RANDOM_MERGE_RESULTS_FILE, "a+") as f:
                            f.write(json.dumps(results) + "\n")

    # Step 6: Plot.
    if args.plot in ["all", "best"]:
        individual_model_db = pd.read_json(INDIVIDUAL_MODEL_RESULTS_FILE, lines=True)
        ood_datasets = [
            "ImageNetV2",
            "ImageNetR",
            "ImageNetSketch",
            "ObjectNet",
            "ImageNetA",
        ]
        available_ood = [d for d in ood_datasets if d in individual_model_db.columns]
        individual_model_db["OOD"] = individual_model_db[available_ood].mean(axis=1)

        uniform_soup_db = pd.read_json(UNIFORM_SOUP_RESULTS_FILE, lines=True)
        uniform_soup_db["OOD"] = uniform_soup_db[available_ood].mean(axis=1)

        greedy_soup_db = pd.read_json(GREEDY_SOUP_RESULTS_FILE, lines=True)
        greedy_soup_db["OOD"] = greedy_soup_db[available_ood].mean(axis=1)

        random_merge_db = pd.read_json(RANDOM_MERGE_RESULTS_FILE, lines=True)
        random_merge_db["OOD"] = random_merge_db[available_ood].mean(axis=1)

        random_merge_db["geo_mean"] = np.sqrt(
            random_merge_db["ImageNet"] * random_merge_db["OOD"]
        )

        plt.figure(figsize=(18, 12))
        ax = plt.gca()

        ax.scatter(
            greedy_soup_db["ImageNet"],
            greedy_soup_db["OOD"],
            marker="*",
            color="C4",
            s=400,
            label="Greedy Soup",
            zorder=10,
        )

        ax.scatter(
            uniform_soup_db["ImageNet"],
            uniform_soup_db["OOD"],
            marker="o",
            color="C0",
            s=200,
            label="Uniform Soup",
            zorder=10,
        )

        ax.scatter(
            individual_model_db["ImageNet"].values[0],
            individual_model_db["OOD"].values[0],
            marker="h",
            color="slategray",
            s=150,
            label="Initialization (LP)",
            zorder=10,
        )

        ax.scatter(
            individual_model_db["ImageNet"].values[1:],
            individual_model_db["OOD"].values[1:],
            marker="d",
            color="C2",
            s=130,
            label="Various hyperparameters",
            zorder=10,
        )

        value_colors = {
            "random": "C1",
            "mean": "C0",
            "median": "pink",
            "max": "purple",
        }
        markers = {False: "o", True: "^"}

        plotted_labels = set()

        if args.plot == "best":
            best_configs = {}

            for _, row in random_merge_db.iterrows():
                params = row["params"]
                value = params["value"]
                elect_sign = params["elect_sign"]
                use_base = params["use_base"]
                alpha = 1.0 if not use_base else params["alpha"]

                key = (value, elect_sign)
                if (
                    key not in best_configs
                    or row["geo_mean"] > best_configs[key]["geo_mean"]
                ):
                    best_configs[key] = row

            plot_data = list(best_configs.values())
        else:
            plot_data = random_merge_db.to_dict("records")

        sorted_data = sorted(
            plot_data,
            key=lambda x: (x["params"]["value"]),
        )

        for row in sorted_data:
            params = row["params"]
            value = params["value"]
            elect_sign = params["elect_sign"]
            use_base = params["use_base"]
            alpha = 1.0 if not use_base else params["alpha"]

            # Construct label
            label_parts = [value.capitalize()]
            if elect_sign:
                label_parts.append("elect sign")
            label_parts.append(f"alpha={alpha:.1f}")

            label = " ".join(
                [
                    label_parts[0],
                    f"({', '.join(label_parts[1:])})" if len(label_parts) > 1 else "",
                ]
            ).strip()

            # Only show label if it hasn't been shown before
            if label in plotted_labels:
                label = None
            else:
                plotted_labels.add(label)

            if use_base:
                color_alpha = 0.4 + (0.5 * alpha) if alpha is not None else 0.65
                ax.scatter(
                    row["ImageNet"],
                    row["OOD"],
                    marker=markers[elect_sign],
                    color=value_colors[value],
                    s=200,
                    label=label,
                    alpha=color_alpha,
                    zorder=5,
                    linewidth=0,
                )
            else:
                ax.scatter(
                    row["ImageNet"],
                    row["OOD"],
                    marker=markers[elect_sign],
                    edgecolors=value_colors[value],
                    facecolors="none",
                    s=200,
                    label=label,
                    linewidth=2,
                    zorder=5,
                )

        ax.set_ylabel("Avg. accuracy on 5 distribution shifts", fontsize=16)
        ax.set_xlabel("ImageNet Accuracy (top-1%)", fontsize=16)
        ax.grid()
        ax.legend(fontsize=13, ncol=2, facecolor="white").set_zorder(100)
        plt.savefig(f"figure_{args.plot}.png", bbox_inches="tight")
