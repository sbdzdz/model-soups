import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict
from merge import state_dict_to_vector
from pathlib import Path


def analyze_model_parameters(model_location, num_models=72):
    """Analyze variance and relative standard deviation of parameters across models."""
    model_paths = [model_location / f"model_{i}.pt" for i in range(num_models)]
    state_dicts = [torch.load(path, map_location="cpu") for path in model_paths]

    layer_stats = defaultdict(dict)
    for layer_name in state_dicts[0].keys():
        layer_params = torch.stack([sd[layer_name].flatten() for sd in state_dicts])
        variance = torch.var(layer_params, dim=0)
        mean = torch.mean(layer_params, dim=0)
        std = torch.sqrt(variance)
        rel_std = torch.abs(
            std / (mean + 1e-8)
        )  # Add small epsilon to prevent division by zero

        layer_stats[layer_name] = {
            "variance": variance.mean().item(),
            "rel_std": rel_std.mean().item(),
            "param_count": layer_params.shape[1],
        }

    # Also compute overall statistics for backward compatibility
    model_vectors = [state_dict_to_vector(sd) for sd in state_dicts]
    model_vectors = torch.stack(model_vectors)
    overall_variance = torch.var(model_vectors, dim=0)
    overall_rel_std = torch.abs(
        torch.sqrt(overall_variance) / torch.mean(model_vectors, dim=0)
    )

    return overall_variance, overall_rel_std, layer_stats


def plot_statistics(variance, rel_std):
    """Create histograms of variances and relative standard deviations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Convert tensors to numpy arrays and flatten
    all_variances = variance.cpu().numpy().flatten()
    all_rel_stds = rel_std.cpu().numpy().flatten()

    # Plot variance histogram
    ax1.hist(np.log10(all_variances), bins=50, color="skyblue", edgecolor="black")
    ax1.set_xlabel("Log10 Variance", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title("Distribution of Parameter Variances", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Plot relative standard deviation histogram
    ax2.hist(all_rel_stds, bins=50, color="lightgreen", edgecolor="black")
    ax2.set_xlabel("Relative Standard Deviation", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Distribution of Relative Standard Deviations", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("parameter_statistics.png", dpi=300, bbox_inches="tight")
    plt.close()


def display_layer_statistics(layer_stats):
    """Display sorted layer-wise statistics."""
    # Sort layers by variance
    sorted_layers = sorted(
        layer_stats.items(), key=lambda x: x[1]["variance"], reverse=True
    )

    print("\nLayer-wise Statistics (sorted by variance):")
    print("-" * 80)
    print(f"{'Layer Name':<40} {'Variance':>12} {'Rel StdDev':>12} {'#Params':>12}")
    print("-" * 80)

    for layer_name, stats in sorted_layers:
        print(
            f"{layer_name:<40} {stats['variance']:>12.2e} {stats['rel_std']:>12.2e} {stats['param_count']:>12,d}"
        )


def create_parameter_difference_heatmap(model_location, num_models=72):
    """Create a heatmap of relative parameter differences (percentage change) compared to model_0."""
    base_model = torch.load(model_location / "model_0.pt", map_location="cpu")
    layer_names = list(base_model.keys())

    # Load and sort models by ImageNet performance
    model_order = []
    with open("individual_model_results.jsonl") as f:
        results = [json.loads(line) for line in f]
    results.sort(key=lambda x: x["ImageNet"])
    model_order = [int(x["model_name"].split("_")[1]) for x in results]

    # Initialize difference matrix with layers as rows and models as columns
    diff_matrix = np.zeros((len(layer_names), num_models))

    # Calculate relative differences for each model and layer
    for col_idx, model_idx in enumerate(model_order):
        model_path = model_location / f"model_{model_idx}.pt"
        result = next(r for r in results if r["model_name"] == f"model_{model_idx}")
        print(f"Processing model {model_idx} with accuracy {result['ImageNet']}")
        curr_model = torch.load(model_path, map_location="cpu")

        for row_idx, layer_name in enumerate(layer_names):
            if model_idx != 0:  # Skip calculation for base model (will remain zeros)
                # Calculate relative difference: (new - old) / |old| * 100
                relative_diff = (
                    (curr_model[layer_name] - base_model[layer_name])
                    / (
                        torch.abs(base_model[layer_name]) + 1e-8
                    )  # Add epsilon to prevent division by zero
                ).abs().mean().item() * 100  # Convert to percentage
                diff_matrix[row_idx, col_idx] = relative_diff

    # Create heatmap with a diverging colormap
    vmax = np.sort(diff_matrix.flatten())[-5]
    plt.figure(figsize=(20, 40))
    im = plt.imshow(diff_matrix, aspect="auto", cmap="viridis", vmax=vmax)
    plt.colorbar(im, label="Relative Difference (%)")

    # Customize axes
    plt.xlabel("Model Index (sorted by ImageNet accuracy)")
    plt.ylabel("Layer Name")

    # Add model numbers on x-axis (rotated for better readability)
    plt.xticks(
        range(len(model_order)), [f"model_{i}" for i in model_order], rotation=90
    )
    # Add layer names on y-axis
    plt.yticks(range(len(layer_names)), layer_names)

    plt.title("Relative Parameter Differences from Base Model (model_0)")
    plt.tight_layout()
    plt.savefig("parameter_differences_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-location",
        type=Path,
        default=Path(os.environ.get("WORK", ".")) / "models",
        help="Path to directory containing model files",
    )
    args = parser.parse_args()
    model_location = args.model_location

    # variances, rel_stds, layer_stats = analyze_model_parameters(model_location)
    # plot_statistics(variances, rel_stds)
    # display_layer_statistics(layer_stats)
    create_parameter_difference_heatmap(model_location)


if __name__ == "__main__":
    main()
