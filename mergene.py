import os
import argparse
import torch
from pathlib import Path
from merge import state_dict_to_vector, vector_to_state_dict
import pygad
import clip
from torchvision.datasets import ImageNet2p
from functools import partial


def main():
    args = parse_args()
    print(f"Loading checkpoints from: {args.models_location}")

    base_model, preprocess = clip.load("ViT-B/32", "cpu", jit=False)
    dataset = ImageNet2p(preprocess, args.data_location, args.batch_size, args.workers)

    checkpoints = load_checkpoints(args.models_location)
    print(f"Loaded {len(checkpoints)} checkpoints")

    task_vectors, base_state_dict = construct_task_vectors(checkpoints)
    print(f"Constructed {len(task_vectors)} task vectors")

    for name in task_vectors.keys():
        print(f"Task vector: {name}")

    fitness_func_with_data = partial(
        fitness_func, dataset=dataset, base_model=base_model
    )


def fitness_func(ga_instance, solution, solution_idx, dataset, base_model):
    """Calculate fitness based on ImageNet2p accuracy.

    Args:
        ga_instance: The genetic algorithm instance
        solution: The candidate solution to evaluate
        solution_idx: Index of the solution in the population
        dataset: The ImageNet2p dataset instance
        base_model: The base CLIP model

    Returns:
        float: The accuracy on ImageNet2p dataset
    """
    model = get_model_from_sd(solution, base_model)
    accuracy = test_model_on_dataset(model, dataset)

    return accuracy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load model checkpoints and construct task vectors"
    )
    parser.add_argument(
        "--models-location",
        type=str,
        default=os.environ.get("WORK", "") + "/models",
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.environ.get("WORK", "") + "/data",
        help="Directory containing ImageNet2p data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for ImageNet2p dataset",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of workers for ImageNet2p dataset",
    )
    return parser.parse_args()


def load_checkpoints(models_dir):
    """Load all checkpoint files from the specified directory."""
    models_dir = Path(models_dir)
    checkpoint_files = list(models_dir.glob("*.pt"))

    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in {models_dir}")

    checkpoints = {}
    for checkpoint_file in checkpoint_files:
        print(f"Loading checkpoint: {checkpoint_file}")
        checkpoints[checkpoint_file.name] = torch.load(
            checkpoint_file, map_location="cpu"
        )

    return checkpoints


def construct_task_vectors(checkpoints):
    """Construct task vectors by subtracting the base model vector."""
    base_model_name = "model_0.pt"
    base_state_dict = checkpoints[base_model_name]

    print(f"Converting base model to vector...")
    base_vector = state_dict_to_vector(base_state_dict)
    task_vectors = {}

    for name, checkpoint in checkpoints.items():
        if name == base_model_name:
            continue

        print(f"Constructing task vector for {name}")

        model_vector = state_dict_to_vector(checkpoint)
        task_vector = model_vector - base_vector
        task_vectors[name] = {
            "vector": task_vector,
            "state_dict": vector_to_state_dict(task_vector, base_state_dict, None),
        }

    return task_vectors, base_state_dict


if __name__ == "__main__":
    main()
