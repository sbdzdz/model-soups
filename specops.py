import argparse
import os
import wget
import torch
import clip
import time
import torch.nn.utils.stateless as stateless
from torch.utils.data import DataLoader
from datasets.imagenet import ImageNet2pShuffled, ImageNet
from utils import ModelWrapper, maybe_dictionarize_batch


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser("~/data"),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model-location",
        type=str,
        default=os.path.expanduser("~/ssd/checkpoints/soups"),
        help="Where to download the models.",
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        default=False,
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
    return parser.parse_args()


class ModernAlphaWrapper(torch.nn.Module):
    def __init__(self, model, parameter_lists):
        super().__init__()
        self.model = model
        self.parameter_lists = parameter_lists

        # Initialize mixing weights
        num_params = len(parameter_lists[0])
        num_models = len(parameter_lists)
        ralpha = torch.ones(num_params, num_models)
        ralpha = torch.nn.functional.softmax(ralpha, dim=1)
        self.alpha_raw = torch.nn.Parameter(ralpha)
        self.beta = torch.nn.Parameter(torch.tensor(1.0))

    def alpha(self):
        return torch.nn.functional.softmax(self.alpha_raw, dim=1)

    def forward(self, x):
        alpha_weights = self.alpha()

        combined_params = {}
        for idx, param_name in enumerate(self.parameter_lists[0].keys()):
            stacked_params = torch.stack(
                [params[param_name] for params in self.parameter_lists], dim=-1
            )

            weights = alpha_weights[idx]
            combined_params[param_name] = (stacked_params @ weights).squeeze()

        output = stateless.functional_call(self.model, combined_params, (x,))

        return self.beta * output


def evaluate_model(model, test_dset, criterion, device):
    model.eval()
    with torch.no_grad():
        correct = 0.0
        n = 0
        total_loss = 0.0

        for i, batch in enumerate(test_dset.test_loader):
            batch = maybe_dictionarize_batch(batch)
            inputs, labels = batch["images"].to(device), batch["labels"].to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            n += labels.size(0)

            if i % 10 == 0:
                print(f"Eval Progress: [{i}/{len(test_dset.test_loader)}]")

        acc = correct / float(n)
        avg_loss = total_loss / len(test_dset.test_loader)
        print(f"Evaluation - Accuracy: {100*acc:.2f}%, Avg Loss: {avg_loss:.4f}")
    return acc


def main():
    args = parse_arguments()
    NUM_MODELS = 72
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download models if needed
    if args.download_models:
        if not os.path.exists(args.model_location):
            os.makedirs(args.model_location)
        for i in range(NUM_MODELS):
            print(f"\nDownloading model {i} of {NUM_MODELS - 1}")
            wget.download(
                f"https://github.com/mlfoundations/model-soups/releases/download/v0.0.2/model_{i}.pt",
                out=args.model_location,
            )

    # Load base model and datasets
    base_model, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)
    train_dset = ImageNet2pShuffled(
        preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    test_dset = ImageNet(
        preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    # Load model checkpoints
    model_paths = [
        os.path.join(args.model_location, f"model_{i}.pt") for i in range(NUM_MODELS)
    ]
    parameter_lists = [torch.load(cp, map_location="cpu") for cp in model_paths]

    # Setup model
    feature_dim = parameter_lists[0]["classification_head.weight"].shape[1]
    num_classes = parameter_lists[0]["classification_head.weight"].shape[0]
    model = ModelWrapper(base_model, feature_dim, num_classes, normalize=True)
    alpha_model = ModernAlphaWrapper(model, parameter_lists).to(device)

    # Training setup
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(alpha_model.parameters(), lr=0.05, weight_decay=0.0)
    epochs = 5

    # Training loop
    for epoch in range(epochs):
        alpha_model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for i, batch in enumerate(train_dset.train_loader):
            batch = maybe_dictionarize_batch(batch)
            inputs, labels = batch["images"].to(device), batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = alpha_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i % 10 == 0:
                percent_complete = 100.0 * i / len(train_dset.train_loader)
                print(
                    f"Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(train_dset.train_loader)}] "
                    f"Loss: {loss.item():.4f} Time: {time.time()-start_time:.2f}s"
                )
                start_time = time.time()

        # Evaluate after each epoch
        print(f"\nEvaluating epoch {epoch}")
        acc = evaluate_model(alpha_model, test_dset, criterion, device)

        # Optional: save checkpoints
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': alpha_model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'accuracy': acc,
        # }, f'checkpoint_epoch_{epoch}.pt')

    # Final evaluation
    print("\nFinal Evaluation:")
    final_acc = evaluate_model(alpha_model, test_dset, criterion, device)
    print(f"Final Accuracy: {100*final_acc:.2f}%")

    # Save final weights
    torch.save(
        {"alpha": alpha_model.alpha(), "beta": alpha_model.beta}, f"alphas_final.pt"
    )


if __name__ == "__main__":
    main()
