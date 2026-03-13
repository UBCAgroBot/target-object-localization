import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryClassifier(nn.Module):
    def __init__(self, device: str | None = None) -> None:
        super().__init__()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Convolutional layers
        # Input: (B, 3, 64, 64)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)
        self.to(self.device)

    def forward(self: "BinaryClassifier", x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (B, 3, 64, 64)

        Returns:
            Logit (raw score) of shape (B, 1)
            Apply sigmoid to get probability: torch.sigmoid(output)
        """
        x = x.to(self.device)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))

        x = x.view(x.size(0), -1)  # Flatten: (B, 256*4*4)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # (B, 1) - raw logit

        return x

    def predict(self: "BinaryClassifier", x: torch.Tensor) -> torch.Tensor:
        """
        Get probability prediction.

        Args:
            x: Input tensor of shape (B, 3, 64, 64)

        Returns:
            Probability that image contains target class (B, 1)
        """
        logit = self.forward(x)
        return torch.sigmoid(logit)


# Dry run testing cases
if __name__ == "__main__":
    # Dry run 1: Random Tensor Input
    # model = BinaryClassifier()
    # B, C, X, Y = 32, 3, 64, 64
    # random_input = torch.randn(B, C, X, Y)
    # output = model(random_input)
    # print(f"Input shape: {random_input.shape}")
    # print(f"Output shape: {output.shape}")

    # Dry run 2: Simple Training
    model = BinaryClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    # Random data
    x = torch.randn(32, 3, 64, 64, device=model.device)
    y = torch.randint(0, 2, (32, 1), device=model.device).float()

    # Train for 150 steps
    for i in range(150):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"Step {i}: Loss = {loss.item():.4f}")

    print(f"Final loss: {loss.item():.4f}")
