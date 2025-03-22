import argparse
import os
from pathlib import Path
import wget
import torch
from torch import func
import clip
import time
from torch.utils.data import DataLoader
from datasets.imagenet import ImageNet2pShuffled, ImageNet
from utils import ModelWrapper, maybe_dictionarize_batch, test_model_on_dataset
import multiprocessing
import wandb


class WeightedMergeLayer(torch.nn.Module):
    def __init__(self, model, checkpoints):
        """
        An interpolation merge with learnable weights (one weight per layer per model).
        """
        super().__init__()
        self.model = model
        self.checkpoints = checkpoints
        num_params = len(checkpoints[0])
        num_models = len(checkpoints)
        alpha = torch.nn.functional.softmax(torch.ones(num_params, num_models), dim=1)
        self._alpha = torch.nn.Parameter(alpha)
        self.beta = torch.nn.Parameter(torch.tensor(1.0))

    @property
    def alpha(self):
        return torch.nn.functional.softmax(self._alpha, dim=1)

    def forward(self, x):
        combined_params = {}
        for idx, param_name in enumerate(self.checkpoints[0].keys()):
            stacked_params = torch.stack(
                [params[param_name] for params in self.checkpoints], dim=-1
            )
            weights = self.alpha[idx].cpu()
            combined_params[param_name] = (
                (stacked_params @ weights).squeeze().to(x.device)
            )

        output = func.functional_call(self.model, combined_params, (x,))

        return self.beta * output


class WeightedMergeModel(torch.nn.Module):
    def __init__(self, model, checkpoints):
        """
        An interpolation merge with learnable weights (one weight per model).
        """
        super().__init__()
        self.model = model
        self.checkpoints = checkpoints
        num_models = len(checkpoints)
        alpha = torch.nn.functional.softmax(torch.ones(num_models), dim=0)
        self._alpha = torch.nn.Parameter(alpha)

    @property
    def alpha(self):
        return torch.nn.functional.softmax(self._alpha, dim=0)

    def forward(self, x):
        combined_params = {}
        weights = self.alpha.cpu()

        for param_name in self.checkpoints[0].keys():
            stacked_params = torch.stack(
                [params[param_name] for params in self.checkpoints], dim=0
            )

            weight_shape = [weights.shape[0]] + [1] * (stacked_params.dim() - 1)
            broadcast_weights = weights.view(*weight_shape)

            combined_params[param_name] = torch.sum(
                stacked_params * broadcast_weights, dim=0
            ).to(x.device)

        return func.functional_call(self.model, combined_params, (x,))


class WeightedMergeSpectrum(torch.nn.Module):
    def __init__(self, model, checkpoints, num_singular_values):
        """
        A merge that modifies the spectrum of parameter matrices by scaling
        the top singular values with learnable parameters.

        Args:
            model: The base model
            checkpoints: List of model checkpoints
            num_singular_values: Number of top singular values to modify
        """
        super().__init__()
        self.model = model
        self.checkpoints = checkpoints
        self.num_singular_values = num_singular_values
        num_params = len(checkpoints[0])

        self.alpha = torch.nn.Parameter(torch.ones(num_params, num_singular_values))

        self.svd_components = {}
        for param_name in self.checkpoints[0].keys():
            stacked_params = torch.stack(
                [params[param_name] for params in self.checkpoints], dim=0
            )
            avg_param = torch.mean(stacked_params, dim=0)

            if len(avg_param.shape) < 2 or min(avg_param.shape) < num_singular_values:
                self.svd_components[param_name] = None
                continue

            original_shape = avg_param.shape
            U, S, Vh = torch.linalg.svd(avg_param, full_matrices=False)
            self.svd_components[param_name] = {
                "U": U,
                "S": S,
                "Vh": Vh,
                "original_shape": original_shape,
            }

    def forward(self, x):
        combined_params = {}

        for idx, param_name in enumerate(self.checkpoints[0].keys()):
            if self.svd_components[param_name] is None:
                stacked_params = torch.stack(
                    [params[param_name] for params in self.checkpoints], dim=0
                )
                combined_params[param_name] = torch.mean(stacked_params, dim=0).to(
                    x.device
                )
                continue

            components = self.svd_components[param_name]
            U, S, Vh = components["U"], components["S"], components["Vh"]

            S_modified = S.clone()
            top_k = min(self.num_singular_values, len(S))
            S_modified[:top_k] = S[:top_k] * self.alpha[idx, :top_k].cpu()

            reconstructed = torch.matmul(U * S_modified.unsqueeze(0), Vh)
            combined_params[param_name] = reconstructed.to(x.device)

        return func.functional_call(self.model, combined_params, (x,))


def main(args):
    wandb.init(
        entity="codis",
        project="specops",
        mode=args.wandb_mode,
        config={
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
        },
    )

    num_models = 72
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)

    train_dataset = ImageNet2pShuffled(
        preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=min(multiprocessing.cpu_count() * 2, 8),
    )
    num_batches = len(train_dataset.train_loader)

    test_dataset = ImageNet(
        preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=min(multiprocessing.cpu_count() * 2, 8),
    )

    model_paths = [args.model_location / f"model_{i}.pt" for i in range(num_models)]

    raw_checkpoints = [torch.load(path, map_location="cpu") for path in model_paths]
    checkpoints = [
        {
            k: v.requires_grad_(False) if isinstance(v, torch.Tensor) else v
            for k, v in checkpoint.items()
        }
        for checkpoint in raw_checkpoints
    ]

    feature_dim = checkpoints[0]["classification_head.weight"].shape[1]
    num_classes = checkpoints[0]["classification_head.weight"].shape[0]
    model = ModelWrapper(base_model, feature_dim, num_classes, normalize=True)

    for param in model.parameters():
        param.requires_grad = False

    if args.weighting == "layer":
        alpha_model = WeightedMergeLayer(model, checkpoints)
    elif args.weighting == "spectrum":
        alpha_model = WeightedMergeSpectrum(
            model, checkpoints, num_singular_values=args.num_singular_values
        )
    else:
        alpha_model = WeightedMergeModel(model, checkpoints)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = create_optimizer(alpha_model, args)

    for epoch in range(args.epochs):
        alpha_model.train()

        epoch_loss = 0.0
        for i, batch in enumerate(train_dataset.train_loader):
            batch = maybe_dictionarize_batch(batch)
            inputs, labels = batch["images"].to(device), batch["labels"].to(device)

            def closure():
                optimizer.zero_grad()
                outputs = alpha_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                return loss

            if args.optimizer == "adam":
                loss = closure()
                optimizer.step()

            elif args.optimizer == "lbfgs":
                loss = optimizer.step(closure)

            log_gradient_norms(alpha_model)
            log_alpha_values(alpha_model, args)

            epoch_loss += loss.item()

            print(f"Epoch {epoch}, {i}/{num_batches}, Loss: {loss.item():.4f}")
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/step": epoch * len(train_dataset.train_loader) + i,
                }
            )

        epoch_loss = epoch_loss / len(train_dataset.train_loader)
        print(f"Epoch {epoch} Average Loss: {epoch_loss:.4f}")

        wandb.log({"train/epoch_loss": epoch_loss, "epoch": epoch})

        if args.eval_every_epoch:
            alpha_model.eval()
            with torch.no_grad():
                train_acc = test_model_on_dataset(alpha_model, train_dataset)
                print(f"Epoch {epoch} Train Accuracy: {train_acc:.2f}%")
                wandb.log({"train/accuracy": train_acc, "epoch": epoch})

    test_acc = test_model_on_dataset(alpha_model, test_dataset)
    print(f"Test Accuracy: {test_acc:.2f}%")
    wandb.log({"test/accuracy": test_acc})

    wandb.finish()


def create_optimizer(alpha_model, args):
    """
    Create the appropriate optimizer based on command line arguments.

    Returns:
        torch.optim.Optimizer: The initialized optimizer
    """
    if args.optimizer == "adam":
        if args.learning_rate is None:
            args.learning_rate = 0.001
        return torch.optim.AdamW(
            alpha_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "lbfgs":
        if args.learning_rate is None:
            args.learning_rate = 1.0
        return torch.optim.LBFGS(
            alpha_model.parameters(),
            lr=args.learning_rate,
            max_iter=20,
            history_size=100,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def log_alpha_values(alpha_model, args):
    """
    Log alpha values (or average alpha values) for each model to wandb.
    """
    if args.weighting == "layer":
        alpha_distributions = alpha_model.alpha
        avg_alpha_per_model = (
            torch.mean(alpha_distributions, dim=0).detach().cpu().numpy()
        )

        alpha_dict = {
            f"alpha_over_time/model_{i}": avg_alpha_per_model[i]
            for i in range(len(avg_alpha_per_model))
        }
    else:
        alpha_values = alpha_model.alpha.detach().cpu().numpy()

        alpha_dict = {
            f"alpha_over_time/model_{i}": alpha_values[i]
            for i in range(len(alpha_values))
        }

    wandb.log(alpha_dict)


def log_gradient_norms(alpha_model):
    """
    Log the gradient norms of alpha and beta parameters to wandb.
    """
    alpha_grad_norm = torch.norm(alpha_model.alpha.grad).item()
    wandb.log(
        {
            "gradients/alpha_norm": alpha_grad_norm,
        }
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=Path,
        default=Path(os.environ["WORK"]) / "datasets",
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model-location",
        type=Path,
        default=Path(os.environ["WORK"]) / "models",
        help="Where to download the models.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate for the optimizer (if None, uses PyTorch defaults: 0.001 for Adam, 1.0 for LBFGS)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for the optimizer",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="Optimizer to use",
    )
    parser.add_argument(
        "--weighting",
        type=str,
        choices=["layer", "model", "spectrum"],
        default="model",
        help="Weighting scheme: 'layer' uses per-layer weights (WeightedMergeLayer), "
        "'model' uses per-model weights (WeightedMergeModel), "
        "'spectrum' scales top singular values (WeightedMergeSpectrum)",
    )
    parser.add_argument(
        "--eval-every-epoch",
        action="store_true",
        help="Evaluate accuracy on training set after every epoch",
    )
    parser.add_argument(
        "--num-singular-values",
        type=int,
        default=100,
        help="Number of top singular values to modify in spectrum weighting",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="online",
        help="Weights & Biases logging mode",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="model-soups",
        help="Weights & Biases project name",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
