import argparse
import os
import wget
import torch
import clip
import os
import json
import operator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets import ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ObjectNet, ImageNetA
from utils import get_model_from_sd, test_model_on_dataset

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model-location",
        type=str,
        default=os.path.expanduser('~/ssd/checkpoints/soups'),
        help="Where to download the models.",
    )
    parser.add_argument(
        "--download-models", action="store_true", default=False,
    )
    parser.add_argument(
        "--eval-individual-models", action="store_true", default=False,
    )
    parser.add_argument(
        "--uniform-soup", action="store_true", default=False,
    )
    parser.add_argument(
        "--greedy-soup", action="store_true", default=False,
    )
    parser.add_argument(
        "--randmerge", action="store_true", default=False,
    )
    parser.add_argument(
        "--plot", action="store_true", default=False,
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
        nargs='+',
        default=[1.0],
        help="List of scaling factors for merged task vectors",
    )
    return parser.parse_args()

def state_dict_to_vector(state_dict, remove_keys):
    """Convert a state dictionary to a flattened vector.

    Args:
        state_dict (dict): The state dictionary to convert.
        remove_keys (list): Keys to remove from the state dictionary before conversion.

    Returns:
        torch.Tensor: A flattened vector representation of the state dictionary.
    """
    shared_state_dict = {
        k: v for k, v in state_dict.items() if k not in remove_keys
    }
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

if __name__ == '__main__':
    args = parse_arguments()
    NUM_MODELS = 72
    INDIVIDUAL_MODEL_RESULTS_FILE = 'individual_model_results.jsonl'
    UNIFORM_SOUP_RESULTS_FILE = 'uniform_soup_results.jsonl'
    GREEDY_SOUP_RESULTS_FILE = 'greedy_soup_results.jsonl'
    RANDOM_MERGE_RESULTS_FILE = 'random_merge_results.jsonl'

    # Step 1: Download models.
    if args.download_models:
        if not os.path.exists(args.model_location):
            os.mkdir(args.model_location)
        for i in range(NUM_MODELS):
            model_path = os.path.join(args.model_location, f'model_{i}.pt')
            if os.path.exists(model_path):
                print(f'\nSkipping model {i} - file already exists')
            else:
                print(f'\nDownloading model {i} of {NUM_MODELS - 1}')
                wget.download(
                    f'https://github.com/mlfoundations/model-soups/releases/download/v0.0.2/model_{i}.pt',
                    out=args.model_location
                    )

    model_paths = [os.path.join(args.model_location, f'model_{i}.pt') for i in range(NUM_MODELS)]

    # Step 2: Evaluate individual models.
    if args.eval_individual_models or args.uniform_soup or args.greedy_soup or args.randmerge:
        base_model, preprocess = clip.load('ViT-B/32', 'cpu', jit=False)

    if args.eval_individual_models:
        if os.path.exists(INDIVIDUAL_MODEL_RESULTS_FILE):
            os.remove(INDIVIDUAL_MODEL_RESULTS_FILE)
        for j, model_path in enumerate(model_paths):
            assert os.path.exists(model_path)
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model = get_model_from_sd(state_dict, base_model)

            results = {'model_name' : f'model_{j}'}
            # Note: ImageNet2p is the held-out minival set from ImageNet train that we use.
            # It is called 2p for 2 percent of ImageNet, or 26k images.
            # See utils on how this dataset is handled slightly differently.
            for dataset_cls in [ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ObjectNet, ImageNetA]:

                print(f'Evaluating model {j} of {NUM_MODELS - 1} on {dataset_cls.__name__}.')

                dataset = dataset_cls(preprocess, args.data_location, args.batch_size, args.workers)
                accuracy = test_model_on_dataset(model, dataset)
                results[dataset_cls.__name__] = accuracy
                print(accuracy)

            with open(INDIVIDUAL_MODEL_RESULTS_FILE, 'a+') as f:
                f.write(json.dumps(results) + '\n')

    # Step 3: Uniform Soup.
    if args.uniform_soup:
        if os.path.exists(UNIFORM_SOUP_RESULTS_FILE):
            os.remove(UNIFORM_SOUP_RESULTS_FILE)

        # create the uniform soup sequentially to not overload memory
        for j, model_path in enumerate(model_paths):

            print(f'Adding model {j} of {NUM_MODELS - 1} to uniform soup.')

            assert os.path.exists(model_path)
            state_dict = torch.load(model_path)
            if j == 0:
                uniform_soup = {k : v * (1./NUM_MODELS) for k, v in state_dict.items()}
            else:
                uniform_soup = {k : v * (1./NUM_MODELS) + uniform_soup[k] for k, v in state_dict.items()}

        model = get_model_from_sd(uniform_soup, base_model)

        results = {'model_name' : f'uniform_soup'}
        for dataset_cls in [ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ObjectNet, ImageNetA]:

            print(f'Evaluating on {dataset_cls.__name__}.')

            dataset = dataset_cls(preprocess, args.data_location, args.batch_size, args.workers)
            accuracy = test_model_on_dataset(model, dataset)
            results[dataset_cls.__name__] = accuracy
            print(accuracy)

        with open(UNIFORM_SOUP_RESULTS_FILE, 'a+') as f:
            f.write(json.dumps(results) + '\n')


    # Step 4: Greedy Soup.
    if args.greedy_soup:
        if os.path.exists(GREEDY_SOUP_RESULTS_FILE):
            os.remove(GREEDY_SOUP_RESULTS_FILE)

        # Sort models by decreasing accuracy on the held-out validation set ImageNet2p
        # (We call the held out-val set ImageNet2p because it is 2 percent of ImageNet train)
        individual_model_db = pd.read_json(INDIVIDUAL_MODEL_RESULTS_FILE, lines=True)
        individual_model_val_accs = {}
        for _, row in individual_model_db.iterrows():
            individual_model_val_accs[row['model_name']] = row['ImageNet2p']
        individual_model_val_accs = sorted(individual_model_val_accs.items(), key=operator.itemgetter(1))
        individual_model_val_accs.reverse()
        sorted_models = [x[0] for x in individual_model_val_accs]

        # Start the soup by using the first ingredient.
        greedy_soup_ingredients = [sorted_models[0]]
        greedy_soup_params = torch.load(os.path.join(args.model_location, f'{sorted_models[0]}.pt'))
        best_val_acc_so_far = individual_model_val_accs[0][1]
        held_out_val_set = ImageNet2p(preprocess, args.data_location, args.batch_size, args.workers)

        # Now, iterate through all models and consider adding them to the greedy soup.
        for i in range(1, NUM_MODELS):
            print(f'Testing model {i} of {NUM_MODELS}')

            # Get the potential greedy soup, which consists of the greedy soup with the new model added.
            new_ingredient_params = torch.load(os.path.join(args.model_location, f'{sorted_models[i]}.pt'))
            num_ingredients = len(greedy_soup_ingredients)
            potential_greedy_soup_params = {
                k : greedy_soup_params[k].clone() * (num_ingredients / (num_ingredients + 1.)) +
                    new_ingredient_params[k].clone() * (1. / (num_ingredients + 1))
                for k in new_ingredient_params
            }

            # Run the potential greedy soup on the held-out val set.
            model = get_model_from_sd(potential_greedy_soup_params, base_model)
            held_out_val_accuracy = test_model_on_dataset(model, held_out_val_set)

            # If accuracy on the held-out val set increases, add the new model to the greedy soup.
            print(f'Potential greedy soup val acc {held_out_val_accuracy}, best so far {best_val_acc_so_far}.')
            if held_out_val_accuracy > best_val_acc_so_far:
                greedy_soup_ingredients.append(sorted_models[i])
                best_val_acc_so_far = held_out_val_accuracy
                greedy_soup_params = potential_greedy_soup_params
                print(f'Adding to soup. New soup is {greedy_soup_ingredients}')

        # Finally, evaluate the greedy soup.
        model = get_model_from_sd(greedy_soup_params, base_model)
        results = {'model_name' : f'greedy_soup'}
        for dataset_cls in [ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ObjectNet, ImageNetA]:
            print(f'Evaluating on {dataset_cls.__name__}.')
            dataset = dataset_cls(preprocess, args.data_location, args.batch_size, args.workers)
            accuracy = test_model_on_dataset(model, dataset)
            results[dataset_cls.__name__] = accuracy
            print(accuracy)

        with open(GREEDY_SOUP_RESULTS_FILE, 'a+') as f:
            f.write(json.dumps(results) + '\n')

    # Step 5: Random Merge
    if args.randmerge:
        if os.path.exists(RANDOM_MERGE_RESULTS_FILE):
            os.remove(RANDOM_MERGE_RESULTS_FILE)

        # Load first model to get parameter structure and create base vector
        base_state_dict = torch.load(model_paths[0])
        remove_keys = []  # Adjust if there are any keys to remove
        base_vector = state_dict_to_vector(base_state_dict, remove_keys)

        # Create random index assignments
        total_params = len(base_vector)
        random_indices = torch.randint(0, NUM_MODELS, (total_params,))

        # Create merged vectors for both approaches
        merged_task_vector = torch.zeros_like(base_vector)
        merged_no_base_vector = torch.zeros_like(base_vector)

        # Process one model at a time to save memory
        for model_idx in range(NUM_MODELS):
            print(f'Processing model {model_idx} of {NUM_MODELS-1}')
            state_dict = torch.load(model_paths[model_idx])
            model_vector = state_dict_to_vector(state_dict, remove_keys)
            task_vector = model_vector - base_vector  # Compute task vector

            # Add to merged vectors where random_indices matches
            mask = (random_indices == model_idx)
            merged_task_vector[mask] = task_vector[mask]
            merged_no_base_vector[mask] = model_vector[mask]

        # Evaluate random merge without base vector
        print(f'Evaluating random merge without base vector')
        merged_state_dict = vector_to_state_dict(merged_no_base_vector, base_state_dict, remove_keys)
        model = get_model_from_sd(merged_state_dict, base_model)
        results = {'model_name': 'random_merge_no_base'}

        for dataset_cls in [ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ObjectNet, ImageNetA]:
            print(f'Evaluating random merge (no base) on {dataset_cls.__name__}.')
            dataset = dataset_cls(preprocess, args.data_location, args.batch_size, args.workers)
            accuracy = test_model_on_dataset(model, dataset)
            results[dataset_cls.__name__] = accuracy
            print(accuracy)

        with open(RANDOM_MERGE_RESULTS_FILE, 'a+') as f:
            f.write(json.dumps(results) + '\n')

        # Create and evaluate models for each alpha (original approach)
        for alpha in args.alpha:
            print(f'Evaluating random merge with alpha={alpha}')
            merged_vector = base_vector + alpha * merged_task_vector
            merged_state_dict = vector_to_state_dict(merged_vector, base_state_dict, remove_keys)

            # Evaluate merged model
            model = get_model_from_sd(merged_state_dict, base_model)
            results = {'model_name': f'random_merge_alpha_{alpha}'}

            for dataset_cls in [ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ObjectNet, ImageNetA]:
                print(f'Evaluating random merge (alpha={alpha}) on {dataset_cls.__name__}.')
                dataset = dataset_cls(preprocess, args.data_location, args.batch_size, args.workers)
                accuracy = test_model_on_dataset(model, dataset)
                results[dataset_cls.__name__] = accuracy
                print(accuracy)

            with open(RANDOM_MERGE_RESULTS_FILE, 'a+') as f:
                f.write(json.dumps(results) + '\n')

    # Step 6: Plot.
    if args.plot:
        individual_model_db = pd.read_json(INDIVIDUAL_MODEL_RESULTS_FILE, lines=True)
        # Filter to only include OOD datasets that exist in the results
        ood_datasets = ['ImageNetV2', 'ImageNetR', 'ObjectNet', 'ImageNetA']
        available_ood = [d for d in ood_datasets if d in individual_model_db.columns]
        individual_model_db['OOD'] = individual_model_db[available_ood].mean(axis=1)

        uniform_soup_db = pd.read_json(UNIFORM_SOUP_RESULTS_FILE, lines=True)
        uniform_soup_db['OOD'] = uniform_soup_db[available_ood].mean(axis=1)

        greedy_soup_db = pd.read_json(GREEDY_SOUP_RESULTS_FILE, lines=True)
        greedy_soup_db['OOD'] = greedy_soup_db[available_ood].mean(axis=1)

        random_merge_db = pd.read_json(RANDOM_MERGE_RESULTS_FILE, lines=True)
        random_merge_db['OOD'] = random_merge_db[available_ood].mean(axis=1)

        fig = plt.figure(constrained_layout=True, figsize=(8, 6))
        ax = fig.subplots()

        ax.scatter(
            greedy_soup_db['ImageNet'],
            greedy_soup_db['OOD'],
            marker='*',
            color='C4',
            s=400,
            label='Greedy Soup',
            zorder=10
        )

        ax.scatter(
            uniform_soup_db['ImageNet'],
            uniform_soup_db['OOD'],
            marker='o',
            color='C0',
            s=200,
            label='Uniform Soup',
            zorder=10
        )

        ax.scatter(
            individual_model_db['ImageNet'].values[0],
            individual_model_db['OOD'].values[0],
            marker='h',
            color='slategray',
            s=150,
            label='Initialization (LP)',
            zorder=10
        )

        ax.scatter(
            individual_model_db['ImageNet'].values[1:],
            individual_model_db['OOD'].values[1:],
            marker='d',
            color='C2',
            s=130,
            label='Various hyperparameters',
            zorder=10
        )

        # Plot each random merge result
        for idx, row in random_merge_db.iterrows():
            alpha = float(row['model_name'].split('_')[-1])  # Extract alpha from model name
            # Create color that gets darker with higher alpha
            color = plt.cm.Blues(0.3 + 0.7 * alpha)  # Maps alpha 0->0.3, 1->1.0 on Blues colormap
            ax.scatter(
                row['ImageNet'],
                row['OOD'],
                marker='P',  # pentagon marker
                color=color,
                s=200,
                label=f'Random Merge (α={alpha})',
                alpha=0.7,  # slight transparency
                zorder=10
            )

        ax.set_ylabel('Avg. accuracy on 5 distribution shifts', fontsize=16)
        ax.set_xlabel('ImageNet Accuracy (top-1%)', fontsize=16)
        ax.grid()
        ax.legend(fontsize=13)
        plt.savefig('figure.png', bbox_inches='tight')