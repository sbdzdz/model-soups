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


class LearnedMerge(torch.nn.Module):
    def __init__(self, model, checkpoints):
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


def main(args):
    wandb.init(
        entity="codis",
        project="specops",
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
    checkpoints = [torch.load(path, map_location="cpu") for path in model_paths]
    feature_dim = checkpoints[0]["classification_head.weight"].shape[1]
    num_classes = checkpoints[0]["classification_head.weight"].shape[0]
    model = ModelWrapper(base_model, feature_dim, num_classes, normalize=True)
    alpha_model = LearnedMerge(model, checkpoints)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        alpha_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    for epoch in range(args.epochs):
        alpha_model.train()

        epoch_loss = 0.0
        for i, batch in enumerate(train_dataset.train_loader):
            batch = maybe_dictionarize_batch(batch)
            inputs, labels = batch["images"].to(device), batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = alpha_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(alpha_model.parameters(), max_norm=1.0)

            optimizer.step()

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

    alpha_distributions = alpha_model.alpha()
    for idx, param_name in enumerate(alpha_model.checkpoints[0].keys()):
        wandb.log(
            {
                f"alpha_distributions/{param_name}": wandb.Histogram(
                    alpha_distributions[idx].detach().cpu().numpy()
                ),
            }
        )

    wandb.log(
        {
            "final/beta": alpha_model.beta.item(),
        }
    )

    torch.save(
        {"alpha": alpha_model.alpha(), "beta": alpha_model.beta}, "alphas_final.pt"
    )

    wandb.finish()


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
        default=0.001,
        help="Learning rate for the optimizer",
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
        "--wandb-project",
        type=str,
        default="model-soups",
        help="Weights & Biases project name",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
