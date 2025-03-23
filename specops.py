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
    def __init__(self, model, checkpoints, unnormalised=False, use_task_vectors=True):
        """
        An interpolation merge with learnable weights (one weight per layer per model).
        """
        super().__init__()
        self.model = model
        self.base_checkpoint = checkpoints[0]

        if use_task_vectors:
            self.task_vectors = []
            for checkpoint in checkpoints[1:]:
                task_vector = {}
                for key in checkpoint.keys():
                    task_vector[key] = checkpoint[key] - self.base_checkpoint[key]
                self.task_vectors.append(task_vector)
            self.checkpoints = self.task_vectors
        else:
            self.checkpoints = checkpoints[1:]

        self.use_task_vectors = use_task_vectors
        self.unnormalised = unnormalised
        num_params = len(self.checkpoints[0])
        num_models = len(self.checkpoints)
        alpha = torch.nn.functional.softmax(torch.ones(num_params, num_models), dim=1)
        self._alpha = torch.nn.Parameter(alpha)
        self.beta = torch.nn.Parameter(torch.tensor(1.0))

    @property
    def alpha(self):
        if self.unnormalised:
            return self._alpha
        else:
            return torch.nn.functional.softmax(self._alpha, dim=1)

    def forward(self, x):
        combined_params = {}
        for idx, param_name in enumerate(self.checkpoints[0].keys()):
            stacked_params = torch.stack(
                [params[param_name] for params in self.checkpoints], dim=-1
            )
            weights = self.alpha[idx].cpu()
            combined = (stacked_params @ weights).squeeze()

            if self.use_task_vectors:
                combined = combined + self.base_checkpoint[param_name]

            combined_params[param_name] = combined.to(x.device)

        output = func.functional_call(self.model, combined_params, (x,))

        return self.beta * output


class WeightedMergeModel(torch.nn.Module):
    def __init__(self, model, checkpoints, unnormalised=False, use_task_vectors=True):
        """
        An interpolation merge with learnable weights (one weight per model).
        """
        super().__init__()
        self.model = model
        self.base_checkpoint = checkpoints[0]

        if use_task_vectors:
            self.task_vectors = []
            for checkpoint in checkpoints[1:]:
                task_vector = {}
                for key in checkpoint.keys():
                    task_vector[key] = checkpoint[key] - self.base_checkpoint[key]
                self.task_vectors.append(task_vector)
            self.checkpoints = self.task_vectors
        else:
            self.checkpoints = checkpoints[1:]

        self.use_task_vectors = use_task_vectors
        self.unnormalised = unnormalised
        num_models = len(self.checkpoints)
        alpha = torch.nn.functional.softmax(torch.ones(num_models), dim=0)
        self._alpha = torch.nn.Parameter(alpha)

    @property
    def alpha(self):
        if self.unnormalised:
            return self._alpha
        else:
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

            combined = torch.sum(stacked_params * broadcast_weights, dim=0)

            if self.use_task_vectors:
                combined = combined + self.base_checkpoint[param_name]

            combined_params[param_name] = combined.to(x.device)

        return func.functional_call(self.model, combined_params, (x,))


class WeightedMergeSpectrum(torch.nn.Module):
    def __init__(self, model, checkpoints, num_singular_values, use_task_vectors=True):
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
        self.base_checkpoint = checkpoints[0]

        if use_task_vectors:
            self.task_vectors = []
            for checkpoint in checkpoints[1:]:
                task_vector = {}
                for key in checkpoint.keys():
                    task_vector[key] = checkpoint[key] - self.base_checkpoint[key]
                self.task_vectors.append(task_vector)
            self.checkpoints = self.task_vectors
        else:
            self.checkpoints = checkpoints[1:]

        self.use_task_vectors = use_task_vectors
        self.num_singular_values = num_singular_values
        num_params = len(self.checkpoints[0])

        self.alpha = torch.nn.Parameter(torch.ones(num_params, num_singular_values))

        self.avg_params = {}
        self.svd_components = {}

        for param_name in self.checkpoints[0].keys():
            stacked_params = torch.stack(
                [params[param_name] for params in self.checkpoints], dim=0
            )
            avg_param = torch.mean(stacked_params, dim=0)
            self.avg_params[param_name] = avg_param

            if len(avg_param.shape) != 2 or "text_projection" in param_name:
                self.svd_components[param_name] = None
                continue

            U, S, Vh = torch.linalg.svd(avg_param, full_matrices=False)
            self.svd_components[param_name] = {
                "U": U,
                "S": S,
                "Vh": Vh,
            }

    def forward(self, x):
        combined_params = {}

        for idx, param_name in enumerate(self.checkpoints[0].keys()):
            param = self.avg_params[param_name]

            if self.svd_components[param_name] is not None:
                components = self.svd_components[param_name]
                U, S, Vh = components["U"], components["S"], components["Vh"]

                S_modified = S.clone()
                top_k = min(self.num_singular_values, len(S))
                S_modified[:top_k] = S[:top_k] * self.alpha[idx, :top_k].cpu()
                S_diag = torch.diag(S_modified)

                param = torch.linalg.multi_dot([U, S_diag, Vh])

            if self.use_task_vectors:
                param = param + self.base_checkpoint[param_name]

            combined_params[param_name] = param.to(x.device)

        return func.functional_call(self.model, combined_params, (x,))


def main(args):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    alphas_save_dir = Path("alphas") / timestamp
    alphas_save_dir.mkdir(exist_ok=True, parents=True)

    wandb.init(
        entity="codis",
        project="specops",
        mode=args.wandb_mode,
        config={
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "alphas_dir": str(alphas_save_dir),
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
        alpha_model = WeightedMergeLayer(
            model, checkpoints, args.unnormalised, args.task_vectors
        )
    elif args.weighting == "spectrum":
        alpha_model = WeightedMergeSpectrum(
            model,
            checkpoints,
            num_singular_values=args.num_singular_values,
            use_task_vectors=args.task_vectors,
        )
    else:
        alpha_model = WeightedMergeModel(
            model, checkpoints, args.unnormalised, args.task_vectors
        )

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
            # log_alpha_values(alpha_model, args)

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

        if (
            args.eval_every_n_epochs is not None
            and epoch % args.eval_every_n_epochs == 0
        ):
            alpha_model.eval()
            with torch.no_grad():
                test_acc = test_model_on_dataset(alpha_model, test_dataset)
                print(f"Epoch {epoch} Test Accuracy: {test_acc:.2f}%")
                wandb.log({"test/accuracy": test_acc, "epoch": epoch})

                alpha_path = alphas_save_dir / f"alphas_{epoch}.pt"
                torch.save(alpha_model.alpha, alpha_path)

    test_acc = test_model_on_dataset(alpha_model, test_dataset)
    print(f"Test Accuracy: {test_acc:.2f}%")
    wandb.log({"test/accuracy": test_acc})

    torch.save(alpha_model.alpha, alphas_save_dir / "alphas.pt")

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
    alpha_values = alpha_model.alpha.detach().cpu().numpy()
    try:
        if args.weighting == "layer" or args.weighting == "spectrum":
            alpha_dict = {
                f"alpha_distribution/layer_{i}": alpha_values[i]
                for i in range(len(alpha_values))
            }
        else:
            alpha_dict = {
                f"alpha_over_time/model_{i}": alpha_values[i]
                for i in range(len(alpha_values))
            }

        wandb.log(alpha_dict)
    except ValueError:
        print("No alpha values to log")


def log_gradient_norms(alpha_model):
    """
    Log the gradient norms of alpha and beta parameters to wandb.
    """
    if alpha_model.alpha.grad is not None:
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
        default=512,
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
        default=20,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "lbfgs"],
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
        "--eval-every-n-epochs",
        type=int,
        default=None,
        help="Evaluate accuracy on training set after every n epochs",
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
    parser.add_argument(
        "--unnormalised",
        action="store_true",
        help="Use unnormalised weights instead of softmax-normalized weights",
    )
    parser.add_argument(
        "--task-vectors",
        action="store_false",
        help="If True, perform merging on task vectors (checkpoint - base) instead of raw checkpoints",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
