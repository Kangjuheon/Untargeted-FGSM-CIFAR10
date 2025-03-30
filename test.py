import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FGSM Untargeted Attack
def fgsm_attack(model, x, label, eps):
    x_adv = x.clone().detach().to(device)
    x_adv.requires_grad = True

    output = model(x_adv)
    loss = F.cross_entropy(output, label.to(device))
    model.zero_grad()
    loss.backward()

    grad_sign = x_adv.grad.data.sign()
    x_adv = x_adv + eps * grad_sign
    x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv.detach()

# Evaluate clean accuracy
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, label in loader:
            x, label = x.to(device), label.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            correct += pred.eq(label).sum().item()
            total += len(x)
    return 100 * correct / total

# Evaluate under FGSM attack
def evaluate_under_attack(model, loader, eps):
    model.eval()
    correct = 0
    total = 0
    for x, label in loader:
        x, label = x.to(device), label.to(device)
        x_adv = fgsm_attack(model, x, label, eps)
        output = model(x_adv)
        pred = output.argmax(dim=1)
        correct += pred.eq(label).sum().item()
        total += len(x)
    return 100 * correct / total

# Main
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # ResNet18
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    # 훈련
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(5):
        total_loss = 0
        for x, label in train_loader:
            x, label = x.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

    # 평가
    clean_acc = evaluate(model, test_loader)
    print(f"\n[Clean Accuracy] {clean_acc:.2f}%")

    eps = 0.03
    adv_acc = evaluate_under_attack(model, test_loader, eps)
    print(f"[FGSM Untargeted Accuracy] eps={eps} → {adv_acc:.2f}%")
