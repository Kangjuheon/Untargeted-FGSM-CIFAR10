import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc

# GPU 설정
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

# 정확도 측정 함수
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, label in tqdm(loader, desc="Clean Evaluation"):
            x, label = x.to(device), label.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            correct += pred.eq(label).sum().item()
            total += len(x)
    return 100 * correct / total

# 공격 정확도 측정
def evaluate_under_attack(model, loader, eps):
    model.eval()
    correct = 0
    total = 0
    for x, label in tqdm(loader, desc="FGSM Attack Evaluation"):
        x, label = x.to(device), label.to(device)
        x_adv = fgsm_attack(model, x, label, eps)
        with torch.no_grad():
            output = model(x_adv)
            pred = output.argmax(dim=1)
            correct += pred.eq(label).sum().item()
            total += len(x)
        torch.cuda.empty_cache()
        gc.collect()
    return 100 * correct / total

# 메인
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    # Pretrained ConvNeXt Tiny
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False  # freeze

    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 10)
    model = model.to(device)

    # Optimizer: only train classifier
    optimizer = optim.Adam(model.classifier[2].parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 학습 진행
    model.train()
    for epoch in range(5):
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for x, label in loop:
            x, label = x.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        print(f"[Epoch {epoch+1}] Average Loss: {total_loss / len(train_loader):.4f}")

    # 평가
    clean_acc = evaluate(model, test_loader)
    print(f"\n [Clean Accuracy] {clean_acc:.2f}%")

    eps = 0.03
    adv_acc = evaluate_under_attack(model, test_loader, eps)
    print(f"\n [FGSM Untargeted Accuracy] eps={eps} → {adv_acc:.2f}%")
