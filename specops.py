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
            combined_params[param_name] = (stacked_params @ weights).squeeze()

        combined_params = {k: v.to(x.device) for k, v in combined_params.items()}
        output = func.functional_call(self.model, combined_params, (x,))

        return self.beta * output


def main(args):
    # Initialize wandb
    wandb.init(
        project="model-soups",
        config={
            "num_models": num_models,
            "batch_size": args.batch_size,
            "learning_rate": 0.05,
            "weight_decay": 0.0,
            "epochs": 5,
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
    optimizer = torch.optim.AdamW(alpha_model.parameters(), lr=0.05, weight_decay=0.0)
    epochs = 5

    for epoch in range(epochs):
        alpha_model.train()

        epoch_loss = 0.0
        for i, batch in enumerate(train_dataset.train_loader):
            batch = maybe_dictionarize_batch(batch)
            inputs, labels = batch["images"].to(device), batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = alpha_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch {epoch}, {i}/{num_batches}, Loss: {loss.item():.4f}")
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/step": epoch * len(train_dataset.train_loader) + i,
                    }
                )

        accuracy = test_model_on_dataset(alpha_model, test_dataset)
        print(f"Epoch {epoch} Accuracy: {100*accuracy:.2f}%")
        wandb.log(
            {
                "val/accuracy": accuracy,
                "val/epoch": epoch,
            }
        )

    final_accuracy = test_model_on_dataset(alpha_model, test_dataset)
    print(f"Final Accuracy: {100*final_accuracy:.2f}%")

    # Log final metrics
    wandb.log(
        {
            "final/accuracy": final_accuracy,
            "final/beta": alpha_model.beta.item(),
        }
    )

    # Log alpha distributions for each layer
    alpha_distributions = alpha_model.alpha()
    for idx, param_name in enumerate(alpha_model.checkpoints[0].keys()):
        wandb.log(
            {
                f"alpha_distributions/{param_name}": wandb.Histogram(
                    alpha_distributions[idx].detach().cpu().numpy()
                )
            }
        )

    # Save model weights
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
        "--wandb-project",
        type=str,
        default="model-soups",
        help="Weights & Biases project name",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
