import logging
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from common.binary_classifier import BinaryClassifier
from common.voc_dataset import PascalVOCDataset

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Device config
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

logger.info(f"Using device: {DEVICE}")


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
            logger.info(f"EarlyStopping Counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module) -> None:
        """Saves model when validation loss decrease."""
        torch.save(model.state_dict(), self.path)
        logger.info(f"Validation loss decreased ({val_loss:.6f}).  Saving model ...")


def load_dataset(
    root_dir: str = "data/VOC2012",
    year: str = "2012",
    split: str = "trainval",
    train_split: float = 0.8,
) -> tuple[
    torch.utils.data.Subset[PascalVOCDataset],
    torch.utils.data.Subset[PascalVOCDataset],
    int,
    int,
]:
    """Load and split the Pascal VOC dataset."""
    full_dataset = PascalVOCDataset(root_dir=root_dir, year=year, split=split)

    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    logger.info(
        f"Dataset loaded: {train_size} training samples, {val_size} validation samples"
    )
    return train_dataset, val_dataset, train_size, val_size


def build_model(
    device: torch.device, learning_rate: float, checkpoint_dir: str, patience: int = 5
) -> tuple[BinaryClassifier, optim.Adam, nn.BCEWithLogitsLoss, EarlyStopping]:
    """Build model, optimizer, criterion, and early stopping."""
    model = BinaryClassifier(device=str(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(
        patience=patience, path=os.path.join(checkpoint_dir, "best_model.pth")
    )

    logger.info(f"Model initialized on {device}")
    return model, optimizer, criterion, early_stopping


def train_model() -> None:
    # Configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    TARGET_CLASS_IDX = 14  # 'person' class
    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Data setup
    train_dataset, val_dataset, train_size, val_size = load_dataset()

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Model setup
    model, optimizer, criterion, early_stopping = build_model(
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        checkpoint_dir=CHECKPOINT_DIR,
        patience=5,
    )

    # training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        # training
        for batch_idx, (images, metadata) in enumerate(train_loader):
            images = images.to(DEVICE)

            # Convert 0-19 labels to Binary (1 if target class, 0 if not)
            labels = metadata["label"].to(DEVICE)
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
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_preds += (preds == binary_labels).sum().item()
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
                images = images.to(DEVICE)
                labels = metadata["label"].to(DEVICE)
                binary_labels = (labels == TARGET_CLASS_IDX).float().view(-1, 1)

                outputs = model(images)
                loss = criterion(outputs, binary_labels)

                val_loss += loss.item() * images.size(0)

                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == binary_labels).sum().item()
                val_total += binary_labels.size(0)

        val_loss = val_loss / val_size
        val_acc = val_correct / val_total

        logger.info(
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
            logger.info("Early stopping triggered.")
            break

    # Load best model weights
    model.load_state_dict(
        torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pth"), map_location=DEVICE)
    )
    logger.info("Training Complete. Best model loaded.")


if __name__ == "__main__":
    train_model()
