import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pathlib import Path
import pandas as pd
import random
import numpy as np
from ViT import ViT

DATA_ROOT = Path(__file__).resolve().parent.parent / "Data" / "CIFAR10"
SAVE_PATH = Path(__file__).resolve().parent / "model" / "vit_cifar10.pth"
LOG_PATH = Path(__file__).resolve().parent / "logs" / "train_log.csv"

BATCH_SIZE = 128
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 2e-2
PATCH_SIZE = 4
EMBED_DIM = 256
NUM_HEADS = 8
MLP_HIDDEN_DIM = 1024
DEPTH = 8
DROP_PATH_RATE = 0.1
EPOCHS = 200
PATIENCE = 40
SEED = 114514
LABEL_SMOOTHING = 0.05
MIXUP_ALPHA = 0.2
GRAD_CLIP_NORM = 1.0
EMA_DECAY = 0.9998
MIN_LR = 1e-7


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def mixup_data(images, labels, alpha, device):
    if alpha <= 0:
        return images, labels, labels, 1.0

    lam = np.random.beta(alpha, alpha)
    indices = torch.randperm(images.size(0), device=device)
    mixed_images = lam * images + (1 - lam) * images[indices]
    labels_a = labels
    labels_b = labels[indices]
    return mixed_images, labels_a, labels_b, lam


def mixup_criterion(criterion, outputs, labels_a, labels_b, lam):
    return lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)


class ModelEMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])


def build_optimizer(model, learning_rate, weight_decay):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if (
            param.ndim == 1
            or name.endswith(".bias")
            or "norm" in name.lower()
            or "cls_token" in name
            or "pos_embedding" in name
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate)
    return optimizer


def build_dataloaders(batch_size):
    train_transform = transforms.Compose(
        [
            # 数据增强只给训练集
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    full_train_dataset = datasets.CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=train_transform
    )

    full_eval_dataset = datasets.CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=eval_transform
    )

    test_dataset = datasets.CIFAR10(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=eval_transform
    )

    train_size = 45000
    val_size = 5000

    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(full_train_dataset), generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_eval_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader


def train(model, train_loader, optimizer, criterion, device, ema=None):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        images, labels_a, labels_b, lam = mixup_data(images, labels, MIXUP_ALPHA, device)
        outputs = model(images)
        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    average_loss = total_loss / total_samples
    return average_loss


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total_samples
    accuracy = correct / total

    return average_loss, accuracy


if __name__ == "__main__":
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = build_dataloaders(BATCH_SIZE)
    model = ViT(
        image_size=32,
        patch_size=PATCH_SIZE,
        in_channels=3,
        num_classes=10,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
        depth=DEPTH,
        dropout=0.1,
        emb_dropout=0.1,
        drop_path_rate=DROP_PATH_RATE,
    )
    model = model.to(device)
    ema_model = ViT(
        image_size=32,
        patch_size=PATCH_SIZE,
        in_channels=3,
        num_classes=10,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
        depth=DEPTH,
        dropout=0.1,
        emb_dropout=0.1,
        drop_path_rate=DROP_PATH_RATE,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = build_optimizer(model, LEARNING_RATE, WEIGHT_DECAY)
    ema = ModelEMA(model, decay=EMA_DECAY)
    ema.copy_to(ema_model)
    warmup_epochs = max(5, int(0.05 * EPOCHS))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.2,
                end_factor=1.0,
                total_iters=warmup_epochs
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=EPOCHS - warmup_epochs,
                eta_min=MIN_LR
            )
        ],
        milestones=[warmup_epochs]
    )
    history = []
    best_val_acc = 0.0
    best_epoch = 0
    no_improve_epochs = 0

    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, device, ema=ema)
        ema.copy_to(ema_model)
        val_loss, val_acc = evaluate(ema_model, val_loader, criterion, device)
        test_loss, test_acc = evaluate(ema_model, test_loader, criterion, device)
        scheduler.step()
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            no_improve_epochs = 0
            torch.save(ema_model.state_dict(), SAVE_PATH)
        else:
            no_improve_epochs += 1
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4%} | "
            f"test_loss={test_loss:.4f} | "
            f"test_acc={test_acc:.4%} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )
        if no_improve_epochs >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch + 1}: "
                f"no val_acc improvement for {PATIENCE} epochs."
            )
            break

    # 用验证集最优模型做最终 test 评估，避免输出最后一轮结果
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    best_test_loss, best_test_acc = evaluate(model, test_loader, criterion, device)

    print(
        f"Test Loss (Best): {best_test_loss:.4f} | "
        f"Test Acc (Best): {best_test_acc:.4%}"
    )
    print(
        f"Best Val Acc: {best_val_acc:.4%} (Epoch {best_epoch}) | "
        f"Saved: {SAVE_PATH}"
    )

    pd.DataFrame(history).to_csv(LOG_PATH, index=False)
