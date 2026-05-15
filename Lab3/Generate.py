import torch
from pathlib import Path
import numpy as np
from PoetryModel import PoetryModel

DATA_PATH = Path(__file__).resolve().parent.parent / "Data" / "tang.npz"
MODEL_PATH = Path(__file__).resolve().parent / "model" / "poetry_model.pth"

EMBEDDING_DIM = 512
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.5
MAX_GEN_LEN = 100
TEMPERATURE = 0.7
TOP_P = 0.9


def load_model(device):
    datas = np.load(DATA_PATH, allow_pickle=True)
    ix2word = datas["ix2word"].item()
    word2ix = datas["word2ix"].item()

    model = PoetryModel(
        vocab_size=len(word2ix),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, ix2word, word2ix


def generate(model, start_words, ix2word, word2ix, device,
             max_len=MAX_GEN_LEN, temperature=TEMPERATURE, top_p=TOP_P):
    model.eval()
    results = list(start_words)
    start_len = len(start_words)
    input_tensor = torch.tensor([word2ix["<START>"]]).view(1, 1).to(device)
    hidden = None

    with torch.no_grad():
        for i in range(max_len):
            output, hidden = model(input_tensor, hidden)

            if i < start_len:
                w = results[i]
                next_idx = word2ix.get(w, word2ix["<START>"])
            else:
                logits = output.data[0]
                # top-p (nucleus) 采样: 按概率累积截断
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                sorted_probs = torch.softmax(sorted_logits / temperature, dim=-1)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                keep = cumsum <= top_p
                keep[0] = True   # 始终保留概率最大的 token
                keep = keep[:cumsum.size(0)]
                filtered_probs = sorted_probs * keep.float()
                filtered_probs /= filtered_probs.sum()
                chosen = torch.multinomial(filtered_probs, num_samples=1).item()
                top_idx = sorted_indices[chosen].item()
                w = ix2word.get(top_idx, "<EOP>")
                results.append(w)
                next_idx = top_idx
                if w == "<EOP>":
                    del results[-1]
                    break

            input_tensor = input_tensor.data.new([next_idx]).view(1, 1)

    return "".join(results)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ix2word, word2ix = load_model(device)
    print(f"Model loaded. Vocab size: {len(word2ix)}")

    print("\n--- 生成示例 ---")
    test_starts = ["湖光秋月两相和", "白日依山尽", "春眠不觉晓"]
    for start in test_starts:
        poem = generate(model, start, ix2word, word2ix, device)
        print(f"  首句: {start}")
        print(f"  续写: {poem}")
        print()
