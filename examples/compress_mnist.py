import torch
import torch.nn as nn
import copy
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch_afterburner import nn


class MNISTDryModel(nn.Module):
    def __init__(self, dry_ratio=0.95, dry_strength=0.001):
        super().__init__()
        self.fc1 = nn.DryingLinearLayer(
            28 * 28,
            512,
            dry_ratio=dry_ratio,
            dry_strength=dry_strength,
            child_layer_names=["fc2"],
        )
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 10)  # Output stays fully wet

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))  # drying applied here
        return self.fc2(x)  # standard output layer


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_loader = DataLoader(
        datasets.MNIST(".", train=True, download=True, transform=transform),
        batch_size=64,
        shuffle=True,
    )
    test_loader = DataLoader(
        datasets.MNIST(".", train=False, transform=transform),
        batch_size=1000,
        shuffle=False,
    )

    model = MNISTDryModel(dry_strength=0.001).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    print(model.fc1.get_dry_metrics())

    for epoch in range(20):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                metrics = model.fc1.get_dry_metrics()
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                    f" Loss: {loss.item():.4f} | L2: {metrics['dry_L2']:.6f} | "
                    f"Dry Spr: {metrics['dry_sparsity']:.3%} | "
                    f"Wet Spr: {metrics['wet_sparsity']:.3%}"
                )

        # === Evaluation ===
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        print(f"Epoch {epoch} - Test Accuracy: {100. * correct / total:.2f}%")
