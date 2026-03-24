import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import pandas as pd
from CNN import CNN

DATA_ROOT =  Path(__file__).resolve().parent.parent/"Data"
SAVE_PATH = Path(__file__).resolve().parent/"model"/"cnn_mnist.pth"
LOG_PATH = Path(__file__).resolve().parent/"logs"/"train_log.csv"
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
HIDDEN_DIM = 128
EPOCHS = 20

def build_dataloaders(batch_size) :
    full_train_dataset = datasets.MNIST(
        root = DATA_ROOT,
        train = True,
        download = True,
        transform = transforms.ToTensor()
    )
    
    test_dataset = datasets.MNIST(
        root = DATA_ROOT,
        train = False,
        download = True,
        transform = transforms.ToTensor()
    )
    
    train_size = 55000
    val_size = 5000
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset,
        [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        shuffle = False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False
    )
    
    return train_loader, val_loader, test_loader

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim = 1)
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
    device = torch.device("cuda")
    train_loader, val_loader, test_loader = build_dataloaders(BATCH_SIZE)
    model = CNN(HIDDEN_DIM)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE)
    history = []
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4%}"
        )
    print(
        f"Test Loss: {test_loss:.4f} | "
        f"Test Acc: {test_acc:.4%}"
    )
    pd.DataFrame(history).to_csv(LOG_PATH, index = False)
    torch.save(model.state_dict(), SAVE_PATH)
    
