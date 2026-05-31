from collections import Counter
from pathlib import Path
import math
import torch

from NMTData import load_processed_data
from TransformerModel import Seq2SeqTransformer

MODEL_PATH = Path(__file__).resolve().parent / "model" / "transformer_nmt.pth"
OUTPUT_PATH = Path(__file__).resolve().parent / "logs" / "test_translation.txt"

EMBED_DIM = 256
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
MAX_LEN = 80


def get_ngrams(tokens, n):	# 提取 n-gram
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(predictions, references, max_n=4, smooth=1.0):	# 计算语料级 BLEU
    clipped_counts = [0 for _ in range(max_n)]
    total_counts = [0 for _ in range(max_n)]
    pred_len = 0
    ref_len = 0

    for pred, ref in zip(predictions, references):
        pred_len += len(pred)
        ref_len += len(ref)
        for n in range(1, max_n + 1):
            pred_ngrams = get_ngrams(pred, n)
            ref_ngrams = get_ngrams(ref, n)
            clipped = {
                gram: min(count, ref_ngrams.get(gram, 0))
                for gram, count in pred_ngrams.items()
            }
            clipped_counts[n - 1] += sum(clipped.values())
            total_counts[n - 1] += max(sum(pred_ngrams.values()), 0)

    precisions = []
    for clipped, total in zip(clipped_counts, total_counts):
        precisions.append((clipped + smooth) / (total + smooth))

    log_precision = sum(math.log(p) for p in precisions) / max_n
    brevity_penalty = 1.0
    if pred_len < ref_len:
        brevity_penalty = math.exp(1 - ref_len / max(pred_len, 1))

    return brevity_penalty * math.exp(log_precision) * 100


def load_model(device, src_vocab, tgt_vocab):	# 加载训练好的 Transformer
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
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


def decode(model, src, src_vocab, tgt_vocab, device, max_len=MAX_LEN):
    model.eval()
    src = src.to(device)
    src_key_padding_mask = src == src_vocab.pad_idx

    with torch.no_grad():
        memory = model.encode(src)	# 先编码源语言句子
        ys = torch.tensor([[tgt_vocab.bos_idx]], device=device)	# 以 <bos> 开始生成

        for _ in range(max_len - 1):
            decoder_output = model.decode(ys, memory, src_key_padding_mask)
            logits = model.generator(decoder_output[:, -1, :])
            next_word = torch.argmax(logits, dim=-1).item()	# 每步选择概率最大的词

            ys = torch.cat(
                [ys, torch.tensor([[next_word]], device=device)],
                dim=1
            )
            if next_word == tgt_vocab.eos_idx:
                break

    return tgt_vocab.decode(ys.squeeze(0).tolist())


def translate_ids(model, src_ids, src_vocab, tgt_vocab, device):	# id 形式源句翻译
    src_tensor = torch.tensor(src_ids).unsqueeze(0)
    return decode(model, src_tensor, src_vocab, tgt_vocab, device)


if __name__ == "__main__":
    data, _, _, src_vocab, tgt_vocab = load_processed_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device, src_vocab, tgt_vocab)	# 恢复最优模型参数

    predictions = []
    for src_ids in data["test_sources"]:
        predictions.append(translate_ids(model, src_ids, src_vocab, tgt_vocab, device))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for tokens in predictions:
            f.write(" ".join(tokens) + "\n")

    bleu4 = corpus_bleu(predictions, data["references"], max_n=4)	# 计算 BLEU-4
    print(f"BLEU-4: {bleu4:.2f}")
