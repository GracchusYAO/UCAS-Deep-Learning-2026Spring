import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import pandas as pd
from PoetryModel import PoetryModel

DATA_PATH = Path(__file__).resolve().parent.parent / "Data" / "tang.npz"   # 唐诗数据集
SAVE_PATH = Path(__file__).resolve().parent / "model" / "poetry_model.pth" # 模型保存路径
LOG_PATH = Path(__file__).resolve().parent / "logs" / "train_log.csv"      # 日志保存路径

BATCH_SIZE = 128
EMBEDDING_DIM = 512
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 60
PATIENCE = 10
GRAD_CLIP_NORM = 1.0
MIN_LR = 1e-6
SEED = 114514
MAX_SEQ_LEN = 125

def prepare_data():
    """加载数据并去除前导 </s>，使训练和生成时序列格式一致。"""
    datas = np.load(DATA_PATH, allow_pickle=True)
    raw_data = datas["data"]
    ix2word = datas["ix2word"].item() # index -> word
    word2ix = datas["word2ix"].item() # word -> index

    start_idx = word2ix["<START>"]    # 起始 token
    eop_idx = word2ix["<EOP>"]        # 结束 token
    pad_idx = word2ix["</s>"]         # padding token

    poems = []
    for i in range(len(raw_data)):
        row = raw_data[i]
        start_pos = int((row == start_idx).argmax())  # 去除 <START> 前面的 padding
        # 找到 <EOP> 的位置，没有则取最后
        eop_positions = row == eop_idx
        if eop_positions.any():
            eop_pos = int(eop_positions.argmax())
            content = row[start_pos:eop_pos + 1]   # 含 <START> 和 <EOP>
        else:
            content = row[start_pos:]               # 无 <EOP>，取到最后
        # 截断到 MAX_SEQ_LEN（超出则用 <EOP> 收尾）
        if len(content) > MAX_SEQ_LEN:
            content = content[:MAX_SEQ_LEN - 1]
            content = np.append(content, eop_idx)

        # 右填充 </s>
        padded = np.full(MAX_SEQ_LEN, pad_idx, dtype=np.int64)  # 先全部填充为 </s>
        padded[:len(content)] = content                         # 再写入真实诗句
        poems.append(padded)

    data = torch.from_numpy(np.stack(poems)).long()  # 转为 LongTensor，供 Embedding 使用
    return data, ix2word, word2ix


def build_dataloaders(data, batch_size):
    num_total = len(data)
    val_size = max(1, int(num_total * 0.1))  # 10% 作为验证集
    train_size = num_total - val_size
    split_generator = torch.Generator().manual_seed(SEED)
    train_data, val_data = torch.utils.data.random_split(
        data, [train_size, val_size],
        generator=split_generator
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True     # 打乱
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, val_loader

def safe_exp(value):
    return float(np.exp(min(value, 50.0)))  # 防止 loss 过大时 ppl 溢出

def count_tokens(targets, pad_idx):
    """统计非 padding token 的数量。"""
    mask = targets != pad_idx
    num_tokens = mask.sum().item()
    return num_tokens

def train_epoch(model, dataloader, optimizer, criterion, device, pad_idx):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    total_grad_norm = 0.0

    for batch in dataloader:
        poems = batch.to(device)
        inputs = poems[:, :-1]     # 输入：第 0 到 n-1 个 token
        targets = poems[:, 1:]     # 预测目标：第 1 到 n 个 token

        outputs, _ = model(inputs)
        loss = criterion(outputs, targets.reshape(-1)) # 忽略 padding 位置的 loss

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=GRAD_CLIP_NORM # 梯度裁剪，避免梯度爆炸
        )
        optimizer.step()

        num_tokens = count_tokens(targets, pad_idx)
        total_loss += loss.item() * num_tokens  # 按非 padding token 数加权
        total_tokens += num_tokens
        total_grad_norm += float(grad_norm)

    average_loss = total_loss / total_tokens
    perplexity = safe_exp(average_loss)
    average_grad_norm = total_grad_norm / len(dataloader)
    return average_loss, perplexity, average_grad_norm

def evaluate(model, dataloader, criterion, device, pad_idx):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            poems = batch.to(device)
            inputs = poems[:, :-1]
            targets = poems[:, 1:]

            outputs, _ = model(inputs)
            loss = criterion(outputs, targets.reshape(-1))

            num_tokens = count_tokens(targets, pad_idx)
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    average_loss = total_loss / total_tokens
    perplexity = safe_exp(average_loss)
    return average_loss, perplexity

def train(model, train_loader, val_loader, optimizer, criterion, scheduler,
          device, epochs, patience, save_path, pad_idx):
    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0

    for epoch in range(epochs):
        train_loss, train_ppl, grad_norm = train_epoch(
            model, train_loader, optimizer, criterion, device, pad_idx
        )
        val_loss, val_ppl = evaluate(
            model, val_loader, criterion, device, pad_idx
        )
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(model.state_dict(), save_path)  # 只保存验证集 loss 最低的模型
        else:
            no_improve += 1

        history_row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_ppl": train_ppl,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "lr": lr,
            "grad_norm": grad_norm,        # 平均梯度范数
            "is_best": is_best,            # 是否刷新最优验证集 loss
        }
        history.append(history_row)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} train_ppl={train_ppl:.1f} "
            f"val_loss={val_loss:.4f} val_ppl={val_ppl:.1f} "
            f"lr={lr:.6f} grad_norm={grad_norm:.3f}"
        )

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return history, best_epoch, best_val_loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, ix2word, word2ix = prepare_data()
    vocab_size = len(word2ix)
    train_loader, val_loader = build_dataloaders(data, BATCH_SIZE)
    ignore_idx = word2ix["</s>"]   # padding 位置不参与 loss
    model = PoetryModel(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        padding_idx=ignore_idx     # padding embedding 不更新
    ).to(device)
    print(f"Model parameters: {count_parameters(model)}")

    criterion = nn.CrossEntropyLoss(ignore_index=ignore_idx)  # 忽略 padding token

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    warmup_epochs = max(3, int(0.05 * EPOCHS))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.2,
                end_factor=1.0,
                total_iters=warmup_epochs  # 前若干轮线性 warmup
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=EPOCHS - warmup_epochs,
                eta_min=MIN_LR             # 最小学习率
            )
        ],
        milestones=[warmup_epochs]
    )

    history, best_epoch, best_val_loss = train(
        model, train_loader, val_loader, optimizer, criterion, scheduler,
        device, EPOCHS, PATIENCE, SAVE_PATH, ignore_idx
    )

    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    print(f"\nBest model at epoch {best_epoch}, val_loss={best_val_loss:.4f}")
    pd.DataFrame(history).to_csv(LOG_PATH, index=False)
