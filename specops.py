import argparse
from pathlib import Path
import wget
import torch
from torch import func
import clip
import time
from torch.utils.data import DataLoader
from datasets.imagenet import ImageNet2pShuffled, ImageNet
from utils import ModelWrapper, maybe_dictionarize_batch
import multiprocessing


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=str(Path.home() / "data"),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model-location",
        type=str,
        default=str(Path.home() / "ssd" / "checkpoints" / "soups"),
        help="Where to download the models.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )
    return parser.parse_args()


class ModernAlphaWrapper(torch.nn.Module):
    def __init__(self, model, checkpoints):
        super().__init__()
        self.model = model
        self.checkpoints = checkpoints

        num_params = len(checkpoints[0])
        num_models = len(checkpoints)
        ralpha = torch.ones(num_params, num_models)
        ralpha = torch.nn.functional.softmax(ralpha, dim=1)
        self.alpha_raw = torch.nn.Parameter(ralpha)
        self.beta = torch.nn.Parameter(torch.tensor(1.0))

    @property
    def alpha(self):
        return torch.nn.functional.softmax(self.alpha_raw, dim=1)

    def forward(self, x):
        alpha_weights = self.alpha()

        combined_params = {}
        for idx, param_name in enumerate(self.checkpoints[0].keys()):
            stacked_params = torch.stack(
                [params[param_name] for params in self.checkpoints], dim=-1
            )

            weights = alpha_weights[idx]
            combined_params[param_name] = (stacked_params @ weights).squeeze()

        output = func.functional_call(self.model, combined_params, (x,))

        return self.beta * output


def evaluate_model(model, dataset, criterion, device):
    model.eval()
    with torch.no_grad():
        correct = 0.0
        n = 0
        total_loss = 0.0

        for i, batch in enumerate(dataset.test_loader):
            batch = maybe_dictionarize_batch(batch)
            inputs, labels = batch["images"].to(device), batch["labels"].to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            n += labels.size(0)

            if i % 10 == 0:
                print(f"Eval Progress: [{i}/{len(dataset.test_loader)}]")

        acc = correct / float(n)
        avg_loss = total_loss / len(dataset.test_loader)
        print(f"Evaluation - Accuracy: {100*acc:.2f}%, Avg Loss: {avg_loss:.4f}")
    return acc


def main():
    args = parse_arguments()
    num_models = 72
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_location = Path(args.model_location)
    data_location = Path(args.data_location)

    base_model, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)
    train_dataset = ImageNet2pShuffled(
        preprocess,
        location=data_location,
        batch_size=args.batch_size,
        num_workers=min(multiprocessing.cpu_count() * 4, 16),
    )
    test_dataset = ImageNet(
        preprocess,
        location=data_location,
        batch_size=args.batch_size,
        num_workers=min(multiprocessing.cpu_count() * 4, 16),
    )

    model_paths = [model_location / f"model_{i}.pt" for i in range(num_models)]
    parameter_lists = [torch.load(cp, map_location="cpu") for cp in model_paths]

    feature_dim = parameter_lists[0]["classification_head.weight"].shape[1]
    num_classes = parameter_lists[0]["classification_head.weight"].shape[0]
    model = ModelWrapper(base_model, feature_dim, num_classes, normalize=True)
    alpha_model = ModernAlphaWrapper(model, parameter_lists).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(alpha_model.parameters(), lr=0.05, weight_decay=0.0)
    epochs = 5

    for epoch in range(epochs):
        alpha_model.train()
        epoch_loss = 0.0
        start_time = time.time()

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
                percent_complete = 100.0 * i / len(train_dataset.train_loader)
                print(
                    f"Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(train_dataset.train_loader)}] "
                    f"Loss: {loss.item():.4f} Time: {time.time()-start_time:.2f}s"
                )
                start_time = time.time()

        acc = evaluate_model(alpha_model, test_dataset, criterion, device)
        print(f"Epoch {epoch} Accuracy: {100*acc:.2f}%")

    final_acc = evaluate_model(alpha_model, test_dataset, criterion, device)
    print(f"Final Accuracy: {100*final_acc:.2f}%")

    torch.save(
        {"alpha": alpha_model.alpha(), "beta": alpha_model.beta}, "alphas_final.pt"
    )


if __name__ == "__main__":
    main()
