import torch
from typing import List, Optional
from utils import get_model_from_sd
import os
from pathlib import Path
import numpy as np


def random_merge(
    model_paths: List[str],
    base_model: torch.nn.Module,
    use_base: bool = False,
    elect_sign: bool = False,
    alpha: Optional[float] = 1.0,
    remove_keys: Optional[List[str]] = None,
    value: str = "random",
) -> torch.nn.Module:
    """Randomly merge multiple models by assigning each parameter to a random source model.

    Args:
        model_paths: List of paths to model checkpoint files
        base_model: Base model architecture to use for loading
        use_base: Whether to use base model + task vectors (True) or direct parameter merge (False)
        elect_sign: Whether to use sign election for merging (not implemented yet)
        alpha: Scaling factor for task vectors when use_base=True (default: 1.0)
        value: Method to compute parameter values from multiple models

    Returns:
        Merged model instance
    """
    cache_path = get_cache_path(value, elect_sign, use_base, alpha)
    if os.path.exists(cache_path):
        print(f"Loading cached merged model from {cache_path}")
        state_dict = torch.load(cache_path, map_location="cpu")
        return get_model_from_sd(state_dict, base_model)

    if remove_keys is None:
        remove_keys = []

    print("Loading all models into memory...")
    state_dicts = [torch.load(path, map_location="cpu") for path in model_paths]
    base_state_dict = state_dicts[0]

    print("Converting models to vectors...")
    model_vectors = [state_dict_to_vector(sd, remove_keys) for sd in state_dicts]
    base_vector = model_vectors[0]

    if use_base:
        model_vectors = [mv - base_vector for mv in model_vectors]

    total_params = len(base_vector)
    num_models = len(model_paths)

    print("Merging models...")
    model_vectors = torch.stack(model_vectors)

    if value == "random":
        random_indices = torch.randint(0, num_models, (total_params,), device="cpu")
        merged_vector = model_vectors[random_indices, torch.arange(total_params)]
    elif value == "mean":
        merged_vector = model_vectors.mean(dim=0)
    elif value == "max":
        merged_vector = model_vectors.abs().max(dim=0)[0] * torch.sign(
            model_vectors.sum(dim=0)
        )
    elif value == "median":
        merged_vector = torch.median(model_vectors, dim=0)[0]
    else:
        raise ValueError(
            f"Unknown value method: {value}. Choose from: random, mean, max, median"
        )

    if elect_sign:
        param_sums = model_vectors.sum(dim=0)
        elected_signs = torch.sign(param_sums)
        merged_vector = merged_vector.abs() * elected_signs

    if use_base:
        merged_vector = base_vector + alpha * merged_vector

    merged_state_dict = vector_to_state_dict(
        merged_vector, base_state_dict, remove_keys
    )

    print(f"Saving merged model to {cache_path}")
    torch.save(merged_state_dict, cache_path)

    if use_base:
        alphas = np.arange(0.1, 1.3, 0.1)
        for alpha in alphas:
            alt_cache_path = Path(get_cache_path(value, elect_sign, use_base, alpha))
            if not alt_cache_path.exists():
                alt_merged_vector = base_vector + alpha * (merged_vector - base_vector)
                alt_merged_state_dict = vector_to_state_dict(
                    alt_merged_vector, base_state_dict, remove_keys
                )
                print(f"Saving additional alpha variant ({alpha}) to {alt_cache_path}")
                torch.save(alt_merged_state_dict, alt_cache_path)

    return get_model_from_sd(merged_state_dict, base_model)


def state_dict_to_vector(state_dict, remove_keys):
    """Convert a state dictionary to a flattened vector.

    Args:
        state_dict (dict): The state dictionary to convert.
        remove_keys (list): Keys to remove from the state dictionary before conversion.

    Returns:
        torch.Tensor: A flattened vector representation of the state dictionary.
    """
    shared_state_dict = {k: v for k, v in state_dict.items() if k not in remove_keys}
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for value in shared_state_dict.values()]
    )


def vector_to_state_dict(vector, state_dict, remove_keys):
    """Convert a flattened vector back to a state dictionary.

    Args:
        vector (torch.Tensor): The flattened vector to convert.
        state_dict (dict): The original state dictionary to use as a reference.
        remove_keys (list): Keys that were removed during the flattening process.

    Returns:
        dict: The reconstructed state dictionary.
    """
    reference_dict = {k: v for k, v in state_dict.items() if k not in remove_keys}

    torch.nn.utils.vector_to_parameters(vector, reference_dict.values())

    if "transformer.shared.weight" in reference_dict:
        shared_weight = reference_dict["transformer.shared.weight"]
        for key in remove_keys:
            reference_dict[key] = shared_weight

    return reference_dict


def get_cache_path(
    value: str, elect_sign: bool, use_base: bool, alpha: Optional[float] = None
) -> str:
    """Generate a standardized cache path for merged models.

    Args:
        value: Merge method (random, mean, max, median)
        elect_sign: Whether sign election was used
        use_base: Whether base model + task vectors were used
        alpha: Scaling factor for task vectors (only used if use_base=True)

    Returns:
        str: Path where the merged model should be cached
    """
    work_dir = os.environ.get("WORK", ".")
    model_dir = Path(work_dir) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    parts = [value]
    if elect_sign:
        parts.append("elect_sign")
    if use_base:
        parts.append("use_base")
        if alpha is not None:
            parts.append(f"alpha_{alpha:.1f}")

    filename = "_".join(parts) + ".pt"
    return str(model_dir / filename)
