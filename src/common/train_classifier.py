print("aeroplane")

import os

import torch

print("1")

import torch.nn as nn

print("2")
import torch.optim as optim

print("3")
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

print("4")
from src.common.voc_dataset import PascalVOCDataset

print("5")

transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
print("6")

train_dataset = PascalVOCDataset(
    root_dir="./data/VOC2012", year="2012", split="train", transform=transform
)

test_dataset = PascalVOCDataset(
    root_dir="./data/VOC2012", year="2012", split="val", transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,  # load data in parallel, can try 4, 8 etc.
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of testing samples: {len(test_dataset)}")


class VOCClassifier(nn.Module):
    def __init__(self, num_classes=20) -> None:
        super(VOCClassifier, self).__init__()  # track stuff

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 6 * 6, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


model = VOCClassifier(num_classes=20).to("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # adjust later


def train_an_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, targets) in enumerate(train_loader):  # explain later
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


def testinggg(model, test_loader, criterion, device):
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


# save checkpoint here, triple check b/c it's it's all generated...


def save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir):
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


def train():
    best_test_loss = float("inf")
    patience_counter: int = 0

    print("Starting training...")

    for epoch in range(1, 50):  # change later
        print(f"Epoch {epoch}/{50}")  # change later

        train_loss, train_acc = train_an_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")

        test_loss, test_acc = testinggg(
            model,
            test_loader,
            criterion,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        print(f"Testing Loss: {test_loss:.4f}, Testing Accuracy: {test_acc:.2f}%")

        if epoch % 5 == 0:  # checkpoint frequency
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
                f"No improvement in test loss. Patience counter: {patience_counter}/10"
            )  # early stopping
            if patience_counter >= 10:
                print("Early stopping triggered.")
                break

        print("\n")

    print("=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    train()
