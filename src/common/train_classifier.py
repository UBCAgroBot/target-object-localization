import os
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.common.voc_dataset import PascalVOCDataset
from src.common.VOCclassifier import VOCClassifier

image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]
image_size = 64

BATCH_SIZE = 32
NUM_OF_EPOCHS = 50
CHECKPOINT_FREQ = 5
EARLY_STOPPING_PATIENCE = 10
NUM_WORKERS = 2
LEARNING_RATE = 0.0005


def load_dataset() -> (
    tuple[
        PascalVOCDataset,
        PascalVOCDataset,
        DataLoader[tuple[torch.Tensor, dict[str, Any]]],
        DataLoader[tuple[torch.Tensor, dict[str, Any]]],
    ]
):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std),
        ]
    )

    train_dataset = PascalVOCDataset(
        root_dir="./data/VOC2012", year="2012", split="train", transform=transform
    )

    test_dataset = PascalVOCDataset(
        root_dir="./data/VOC2012", year="2012", split="val", transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of testing samples: {len(test_dataset)}")

    return train_dataset, test_dataset, train_loader, test_loader


def train_an_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        labels = targets["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(
                f"Batch {batch_idx + 1}/{len(train_loader)}]"
                f"Loss: {running_loss / (batch_idx + 1):.4f}"
                f"Accuracy: {100. * correct / total:.2f}%"
            )

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def testing(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            labels = targets["label"].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = test_loss / len(test_loader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    val_loss: float,
    checkpoint_dir: str,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    # creates checkpoints folder, and don't error if folder alrd exists

    checkpoint = {  # dictionary to save
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")


def train() -> None:
    train_dataset, test_dataset, train_loader, test_loader = load_dataset()
    model = VOCClassifier(num_classes=20).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_test_loss = float("inf")
    patience_counter: int = 0

    print("Starting training...")

    for epoch in range(1, NUM_OF_EPOCHS):
        print(f"Epoch {epoch}/{NUM_OF_EPOCHS}")

        train_loss, train_acc = train_an_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")

        test_loss, test_acc = testing(
            model,
            test_loader,
            criterion,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        print(f"Testing Loss: {test_loss:.4f}, Testing Accuracy: {test_acc:.2f}%")

        if epoch % CHECKPOINT_FREQ == 0:  # checkpoint frequency
            save_checkpoint(
                model, optimizer, epoch, test_loss, checkpoint_dir="checkpoints"
            )

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0

            save_checkpoint(
                model, optimizer, epoch, test_loss, checkpoint_dir="checkpoints"
            )
            print("New best model found and saved!")

        else:
            patience_counter += 1
            print(
                f"No improvement in test loss. Patience counter: {patience_counter}/{EARLY_STOPPING_PATIENCE}"
            )  # early stopping
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print("\n")

    print("=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    train()
