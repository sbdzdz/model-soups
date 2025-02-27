import torch
from typing import List, Optional
from utils import get_model_from_sd
import os
from pathlib import Path
import numpy as np
import re


def random_merge(
    model_paths: List[str],
    base_model: torch.nn.Module,
    use_base: bool = False,
    elect_sign: bool = False,
    alpha: Optional[float] = 1.0,
    pattern: Optional[str] = None,
    value: str = "random",
) -> torch.nn.Module:
    """Randomly merge multiple models by assigning each parameter to a random source model.

    Args:
        model_paths: List of paths to model checkpoint files
        base_model: Base model architecture to use for loading
        use_base: Whether to use base model + task vectors (True) or direct parameter merge (False)
        elect_sign: Whether to use sign election for merging (not implemented yet)
        alpha: Scaling factor for task vectors when use_base=True (default: 1.0)
        pattern: If provided, only merge parameters whose keys match this regex pattern
        value: Method to compute parameter values from multiple models

    Returns:
        Merged model instance
    """
    cache_path = get_cache_path(value, elect_sign, use_base, alpha)
    if os.path.exists(cache_path):
        print(f"Loading cached merged model from {cache_path}")
        state_dict = torch.load(cache_path, map_location="cpu")
        return get_model_from_sd(state_dict, base_model)

    print("Loading all models into memory...")
    state_dicts = [torch.load(path, map_location="cpu") for path in model_paths]
    base_state_dict = state_dicts[0]

    print("Converting models to vectors...")
    model_vectors = [state_dict_to_vector(sd, pattern) for sd in state_dicts]
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
        max_abs_indices = model_vectors.abs().max(dim=0)[1]
        merged_vector = torch.gather(
            model_vectors, 0, max_abs_indices.unsqueeze(0)
        ).squeeze(0)
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

    merged_state_dict = vector_to_state_dict(merged_vector, base_state_dict, pattern)

    print(f"Saving merged model to {cache_path}")
    torch.save(merged_state_dict, cache_path)

    return get_model_from_sd(merged_state_dict, base_model)


def state_dict_to_vector(state_dict, pattern=None):
    """Convert a state dictionary to a flattened vector.

    Args:
        state_dict (dict): The state dictionary to convert.
        pattern (str): Regex pattern to filter keys in the state dictionary.

    Returns:
        torch.Tensor: A flattened vector representation of the state dictionary.
    """
    if pattern is not None:
        regex = re.compile(pattern)
        shared_state_dict = {k: v for k, v in state_dict.items() if regex.search(k)}
    else:
        shared_state_dict = state_dict

    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for value in shared_state_dict.values()]
    )


def vector_to_state_dict(vector, state_dict, pattern=None):
    """Convert a flattened vector back to a state dictionary.

    Args:
        vector (torch.Tensor): The flattened vector to convert.
        state_dict (dict): The original state dictionary to use as a reference.
        pattern (str): Regex pattern used to filter keys during flattening.

    Returns:
        dict: The reconstructed state dictionary.
    """
    result_dict = {k: v.clone() for k, v in state_dict.items()}

    if pattern is not None:
        regex = re.compile(pattern)
        filtered_values = [v for k, v in result_dict.items() if regex.search(k)]
    else:
        filtered_values = result_dict.values()

    torch.nn.utils.vector_to_parameters(vector, filtered_values)

    return result_dict


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
