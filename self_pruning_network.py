"""
Self-Pruning Neural Network on CIFAR-10
========================================
Tredence AI Engineering Intern – Case Study

Architecture:
  - PrunableLinear layers: each weight has a learnable gate (sigmoid-gated).
  - Total Loss = CrossEntropyLoss + λ * L1(gates)
  - Three λ values compared: 1e-4 (low), 1e-3 (medium), 5e-3 (high)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")           # headless – no display needed
import matplotlib.pyplot as plt
import numpy as np
import os, time

# ──────────────────────────────────────────────
# 1. PrunableLinear Layer
# ──────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that multiplies each weight by a
    learnable gate in [0, 1].  The gate is obtained by passing a raw
    'gate_scores' parameter through a Sigmoid.  Gradients flow through both
    the weight and gate_scores parameters via standard autograd.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight & bias (identical initialisation to nn.Linear)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # One raw gate score per weight – same shape as weight
        # Initialised near 0 so sigmoid(0) ≈ 0.5  (all gates half-open at start)
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Kaiming uniform init for the weight (mirrors nn.Linear default)
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gates ∈ (0, 1)  – differentiable w.r.t. gate_scores
        gates = torch.sigmoid(self.gate_scores)

        # Element-wise mask: prune weights whose gate → 0
        pruned_weights = self.weight * gates

        # Standard affine transform: y = x W^T + b
        return F.linear(x, pruned_weights, self.bias)

    def sparsity_loss(self) -> torch.Tensor:
        """L1 penalty on the current gate values (all positive by construction)."""
        return torch.sigmoid(self.gate_scores).sum()

    def sparsity_level(self, threshold: float = 1e-2) -> float:
        """Fraction of weights whose gate value < threshold (treated as pruned)."""
        gates = torch.sigmoid(self.gate_scores).detach()
        pruned = (gates < threshold).float().mean().item()
        return pruned

    def get_gate_values(self) -> np.ndarray:
        return torch.sigmoid(self.gate_scores).detach().cpu().numpy().ravel()


# ──────────────────────────────────────────────
# 2. Network Definition
# ──────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Simple feed-forward classifier for CIFAR-10 (32×32×3 = 3 072 inputs, 10 classes).
    All linear layers use PrunableLinear.
    """

    def __init__(self):
        super().__init__()
        # Conv stem to reduce spatial dimensions before the FC head
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                    # 16×16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                    # 8×8
        )
        # Prunable fully-connected head
        self.fc1 = PrunableLinear(64 * 8 * 8, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)          # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sparsity_loss(self) -> torch.Tensor:
        """Aggregate L1 gate penalty across all PrunableLinear layers."""
        return self.fc1.sparsity_loss() + self.fc2.sparsity_loss() + self.fc3.sparsity_loss()

    def prunable_layers(self):
        return [self.fc1, self.fc2, self.fc3]


# ──────────────────────────────────────────────
# 3. Data Loading
# ──────────────────────────────────────────────

def get_cifar10_loaders(batch_size: int = 128, data_dir: str = "./data"):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
    test_set  = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=256,        shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ──────────────────────────────────────────────
# 4. Training Loop
# ──────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, lam: float, device):
    model.train()
    total_loss = total_cls_loss = total_spar_loss = correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)

        cls_loss  = F.cross_entropy(logits, labels)
        spar_loss = model.sparsity_loss()
        loss      = cls_loss + lam * spar_loss

        loss.backward()
        optimizer.step()

        total_loss      += loss.item()
        total_cls_loss  += cls_loss.item()
        total_spar_loss += spar_loss.item()
        preds    = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    n = len(loader)
    return (total_loss / n, total_cls_loss / n, total_spar_loss / n, correct / total)


# ──────────────────────────────────────────────
# 5. Evaluation
# ──────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds   = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total


def compute_sparsity(model, threshold: float = 1e-2) -> float:
    """Overall sparsity across all PrunableLinear layers (fraction pruned)."""
    pruned = total = 0
    for layer in model.prunable_layers():
        gates = torch.sigmoid(layer.gate_scores).detach()
        pruned += (gates < threshold).sum().item()
        total  += gates.numel()
    return pruned / total


def get_all_gate_values(model) -> np.ndarray:
    vals = []
    for layer in model.prunable_layers():
        vals.append(layer.get_gate_values())
    return np.concatenate(vals)


# ──────────────────────────────────────────────
# 6. Main Experiment
# ──────────────────────────────────────────────

def run_experiment(lam: float, epochs: int, device, train_loader, test_loader):
    print(f"\n{'='*55}")
    print(f"  Training with λ = {lam}")
    print(f"{'='*55}")

    model     = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, cls_l, spar_l, tr_acc = train_one_epoch(
            model, train_loader, optimizer, lam, device)
        scheduler.step()
        elapsed = time.time() - t0

        if epoch % 5 == 0 or epoch == 1:
            test_acc = evaluate(model, test_loader, device)
            sparsity = compute_sparsity(model)
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"Loss {tr_loss:.4f} (cls {cls_l:.4f}, spar {spar_l:.1f}) | "
                f"Train {tr_acc*100:.1f}% | Test {test_acc*100:.1f}% | "
                f"Sparsity {sparsity*100:.1f}% | {elapsed:.1f}s"
            )

    final_test_acc = evaluate(model, test_loader, device)
    final_sparsity = compute_sparsity(model)
    gate_vals      = get_all_gate_values(model)

    print(f"\n  ✓ Final Test Accuracy : {final_test_acc*100:.2f}%")
    print(f"  ✓ Final Sparsity      : {final_sparsity*100:.2f}%")

    return final_test_acc, final_sparsity, gate_vals


# ──────────────────────────────────────────────
# 7. Plot: Gate-value distribution
# ──────────────────────────────────────────────

def plot_gate_distribution(gate_vals: np.ndarray, lam: float, save_path: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(gate_vals, bins=100, color="#2563EB", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Gate Value (sigmoid output)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Gate-Value Distribution  (λ = {lam})", fontsize=13, fontweight="bold")
    ax.axvline(0.01, color="red", linestyle="--", linewidth=1.2, label="Prune threshold (0.01)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {save_path}")


# ──────────────────────────────────────────────
# 8. Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS   = 30          # increase to 50–60 for better accuracy
    BATCH    = 128
    LAMBDAS  = [1e-4, 1e-3, 5e-3]   # low, medium, high

    print(f"Device : {DEVICE}")
    train_loader, test_loader = get_cifar10_loaders(batch_size=BATCH)

    results   = {}
    best_lam  = None
    best_gate = None
    best_acc  = 0.0

    for lam in LAMBDAS:
        acc, sparsity, gates = run_experiment(
            lam, EPOCHS, DEVICE, train_loader, test_loader)
        results[lam] = {"accuracy": acc, "sparsity": sparsity, "gates": gates}
        if acc > best_acc:
            best_acc  = acc
            best_lam  = lam
            best_gate = gates

    # Save gate-distribution plot for the best model
    os.makedirs("outputs", exist_ok=True)
    plot_gate_distribution(best_gate, best_lam, "outputs/gate_distribution.png")

    # Print summary table
    print("\n\n" + "="*55)
    print("  RESULTS SUMMARY")
    print("="*55)
    print(f"  {'Lambda':<10} {'Test Acc':>10} {'Sparsity':>12}")
    print(f"  {'-'*34}")
    for lam, r in results.items():
        print(f"  {lam:<10} {r['accuracy']*100:>9.2f}% {r['sparsity']*100:>11.2f}%")
    print("="*55)
    print(f"\n  Best model: λ = {best_lam}  (Test Acc = {best_acc*100:.2f}%)")
