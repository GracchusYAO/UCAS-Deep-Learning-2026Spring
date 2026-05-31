import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path
import random
import numpy as np
import pandas as pd

from NMTData import (
    collate_fn,
    load_processed_data,
)
from TransformerModel import Seq2SeqTransformer, count_parameters

SAVE_PATH = Path(__file__).resolve().parent / "model" / "transformer_nmt.pth"
LOG_PATH = Path(__file__).resolve().parent / "logs" / "train_log.csv"

BATCH_SIZE = 64
EMBED_DIM = 256
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 60
PATIENCE = 10
GRAD_CLIP_NORM = 1.0
LABEL_SMOOTHING = 0.1
MIN_LR = 1e-5
LR_DECAY_FACTOR = 0.5
LR_DECAY_PATIENCE = 3
MAX_LEN = 80
SEED = 114514


def set_seed(seed):	# 固定随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(train_dataset, dev_dataset, src_pad_idx, tgt_pad_idx):	# 构造训练和验证 DataLoader
    pad_collate_fn = partial(
        collate_fn,
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=pad_collate_fn
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=pad_collate_fn
    )
    return train_loader, dev_loader


def safe_exp(value):	# 防止 ppl 溢出
    return float(np.exp(min(value, 50.0)))


def count_tokens(targets, pad_idx):	# 统计非 padding token 数
    return (targets != pad_idx).sum().item()


def train_epoch(model, dataloader, optimizer, criterion, device, tgt_pad_idx):	# 单轮训练
    model.train()
    total_loss = 0.0
    total_tokens = 0
    total_grad_norm = 0.0

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:, :-1]	# decoder 输入不含最后一个 token
        tgt_out = tgt[:, 1:]	# 预测目标不含 <bos>

        outputs = model(src, tgt_input)
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(	# 梯度裁剪防止训练不稳定
            model.parameters(), max_norm=GRAD_CLIP_NORM
        )
        optimizer.step()

        num_tokens = count_tokens(tgt_out, tgt_pad_idx)
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        total_grad_norm += float(grad_norm)

    average_loss = total_loss / total_tokens
    perplexity = safe_exp(average_loss)
    average_grad_norm = total_grad_norm / len(dataloader)
    return average_loss, perplexity, average_grad_norm


def evaluate(model, dataloader, criterion, device, tgt_pad_idx):	# 验证集 loss 和 ppl
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]	# 与训练阶段保持同样的右移方式
            tgt_out = tgt[:, 1:]
            outputs = model(src, tgt_input)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt_out.reshape(-1))

            num_tokens = count_tokens(tgt_out, tgt_pad_idx)
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    average_loss = total_loss / total_tokens
    perplexity = safe_exp(average_loss)
    return average_loss, perplexity


def train(model, train_loader, dev_loader, optimizer, criterion, scheduler,
          device, epochs, patience, save_path, tgt_pad_idx):
    history = []
    best_dev_loss = float("inf")
    best_epoch = 0
    no_improve = 0

    for epoch in range(epochs):
        train_loss, train_ppl, grad_norm = train_epoch(
            model, train_loader, optimizer, criterion, device, tgt_pad_idx
        )
        dev_loss, dev_ppl = evaluate(
            model, dev_loader, criterion, device, tgt_pad_idx
        )
        scheduler.step(dev_loss)	# 验证集停滞后再降低学习率
        lr = optimizer.param_groups[0]["lr"]

        is_best = dev_loss < best_dev_loss
        if is_best:
            best_dev_loss = dev_loss
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(model.state_dict(), save_path)	# 只保存验证集最优模型
        else:
            no_improve += 1

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_ppl": train_ppl,
            "dev_loss": dev_loss,
            "dev_ppl": dev_ppl,
            "lr": lr,
            "grad_norm": grad_norm,
            "is_best": is_best,
        }
        history.append(row)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} train_ppl={train_ppl:.1f} "
            f"dev_loss={dev_loss:.4f} dev_ppl={dev_ppl:.1f} "
            f"lr={lr:.6f} grad_norm={grad_norm:.3f}"
        )

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return history, best_epoch, best_dev_loss


if __name__ == "__main__":
    set_seed(SEED)
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, train_dataset, dev_dataset, src_vocab, tgt_vocab = load_processed_data()

    train_loader, dev_loader = build_dataloaders(
        train_dataset,
        dev_dataset,
        src_vocab.pad_idx,
        tgt_vocab.pad_idx
    )

    model = Seq2SeqTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        src_pad_idx=src_vocab.pad_idx,
        tgt_pad_idx=tgt_vocab.pad_idx,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_len=MAX_LEN,
    ).to(device)
    print(f"Src vocab: {len(src_vocab)}, Tgt vocab: {len(tgt_vocab)}")
    print(f"Model parameters: {count_parameters(model)}")

    criterion = nn.CrossEntropyLoss(	# label smoothing 提高泛化
        ignore_index=tgt_vocab.pad_idx,
        label_smoothing=LABEL_SMOOTHING
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(	# 根据 dev loss 调整学习率
        optimizer,
        mode="min",
        factor=LR_DECAY_FACTOR,
        patience=LR_DECAY_PATIENCE,
        min_lr=MIN_LR
    )

    history, best_epoch, best_dev_loss = train(
        model, train_loader, dev_loader, optimizer, criterion, scheduler,
        device, EPOCHS, PATIENCE, SAVE_PATH, tgt_vocab.pad_idx
    )
    pd.DataFrame(history).to_csv(LOG_PATH, index=False)
    print(f"Best model at epoch {best_epoch}, dev_loss={best_dev_loss:.4f}")
