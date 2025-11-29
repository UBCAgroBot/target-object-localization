import os
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from common.binary_classifier import BinaryClassifier
from common.voc_dataset import PascalVOCDataset


class EarlyStopping:
    def __init__(
        self, patience: int = 5, min_delta: float = 0.0, path: str = "best_model.pth"
    ) -> None:
        self.patience = patience  # How many epochs to wait
        self.min_delta = min_delta  # Minimum change to quality as "improvement"
        self.path = path  # Path for the checkpoint to be saved
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping Counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module) -> None:
        """Saves model when validation loss decrease."""
        torch.save(model.state_dict(), self.path)
        print(
            f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ..."
        )


def train_model() -> None:
    # Configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    TARGET_CLASS_IDX = 14  # 'person' class
    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Data Setup
    full_dataset = PascalVOCDataset(
        root_dir="data/VOC2012", year="2012", split="trainval"
    )

    # 80 train / 20 val split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    model = BinaryClassifier(device=str(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(
        patience=5, path=os.path.join(CHECKPOINT_DIR, "best_model.pth")
    )

    # training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        # training
        for batch_idx, (images, metadata) in enumerate(train_loader):
            images = images.to(device)

            # Convert 0-19 labels to Binary (1 if target class, 0 if not)
            labels = metadata["label"].to(device)
            binary_labels = (labels == TARGET_CLASS_IDX).float().view(-1, 1)

            # Forward
            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, binary_labels)

            # Backward
            loss.backward()
            optimizer.step()

            # Metrics
            running_loss += loss.item() * images.size(0)

            # Calculate accuracy (sigmoid > 0.5 is a positive prediction)
            preds = torch.sigmoid(outputs) > 0.5
            correct_preds += (preds == binary_labels.byte()).sum().item()
            total_preds += binary_labels.size(0)

        epoch_loss = running_loss / train_size
        epoch_acc = correct_preds / total_preds

        # validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, metadata in val_loader:
                images = images.to(device)
                labels = metadata["label"].to(device)
                binary_labels = (labels == TARGET_CLASS_IDX).float().view(-1, 1)

                outputs = model(images)
                loss = criterion(outputs, binary_labels)

                val_loss += loss.item() * images.size(0)

                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == binary_labels.byte()).sum().item()
                val_total += binary_labels.size(0)

        val_loss = val_loss / val_size
        val_acc = val_correct / val_total

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
            f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        # checkpointing
        if (epoch + 1) % 10 == 0:  # Saves every 10 epochs
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": val_loss,
                },
                checkpoint_path,
            )

        # early stopping
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Load best model weights
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pth")))
    print("Training Complete. Best model loaded.")


if __name__ == "__main__":
    train_model()
