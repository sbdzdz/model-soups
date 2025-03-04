import os
import argparse
import torch
from pathlib import Path
from merge import state_dict_to_vector, vector_to_state_dict


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

    if base_model_name not in checkpoints:
        raise ValueError(f"Base model {base_model_name} not found in checkpoints")

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


def main():
    args = parse_args()
    print(f"Loading checkpoints from: {args.models_location}")

    checkpoints = load_checkpoints(args.models_location)
    print(f"Loaded {len(checkpoints)} checkpoints")

    task_vectors, base_state_dict = construct_task_vectors(checkpoints)
    print(f"Constructed {len(task_vectors)} task vectors")

    for name in task_vectors.keys():
        print(f"Task vector: {name}")


if __name__ == "__main__":
    main()
