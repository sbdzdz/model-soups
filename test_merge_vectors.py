import torch
import numpy as np
from merge import merge_vectors


def test_merge_vectors():
    # Create 5 small model vectors with 10 elements each
    torch.manual_seed(42)  # For reproducibility
    num_models = 5
    vector_size = 10

    # Generate random model vectors
    model_vectors = torch.randn(num_models, vector_size)

    print("Original model vectors:")
    for i, vector in enumerate(model_vectors):
        # Round to 3 decimal places
        rounded_vector = [round(val.item(), 3) for val in vector]
        print(f"Model {i+1}: {rounded_vector}")

    print("\nMerged vectors using different methods:")

    methods = ["random", "mean", "max", "median"]
    elect_sign_options = [False, True]

    for method in methods:
        for elect_sign in elect_sign_options:
            merged = merge_vectors(model_vectors, method, elect_sign)
            # Round to 3 decimal places
            rounded_merged = [round(val.item(), 3) for val in merged]
            print(
                f"{method.capitalize()} merge (elect_sign={elect_sign}): {rounded_merged}"
            )

    print("\nFor max merge, parameters came from these models:")
    max_abs_indices = model_vectors.abs().max(dim=0)[1]
    for i in range(vector_size):
        model_idx = max_abs_indices[i].item()
        param_value = model_vectors[model_idx, i].item()
        print(f"Parameter {i}: from Model {model_idx+1} (value: {param_value:.3f})")


if __name__ == "__main__":
    test_merge_vectors()
