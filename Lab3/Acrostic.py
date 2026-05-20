import torch
import sys
from Generate import load_model

LINE_LEN = 7       # 默认七言，让输出稳定
TEMPERATURE = 0.7
TOP_P = 0.9

SPECIAL_WORDS = {"<START>", "<EOP>", "</s>"}
PUNCTUATIONS = {"，", "。"}


def sample_word(logits, ix2word, word2ix, temperature=TEMPERATURE, top_p=TOP_P):
    logits = logits.clone()
    for word in SPECIAL_WORDS | PUNCTUATIONS:
        if word in word2ix:
            logits[word2ix[word]] = -float("inf")  # 句内不生成特殊token和标点

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits / temperature, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    keep = cumsum <= top_p
    keep[0] = True
    filtered_probs = sorted_probs * keep.float()
    filtered_probs /= filtered_probs.sum()

    chosen = torch.multinomial(filtered_probs, num_samples=1).item()
    index = sorted_indices[chosen].item()
    return index, ix2word.get(index, "")


def feed_word(model, word_idx, hidden, device):
    input_tensor = torch.tensor([word_idx]).view(1, 1).to(device)
    output, hidden = model(input_tensor, hidden)
    return output.data[0], hidden


def generate_acrostic(model, heads, ix2word, word2ix, device,
                      line_len=LINE_LEN, temperature=TEMPERATURE, top_p=TOP_P):
    model.eval()
    for word in heads:
        if word not in word2ix:
            raise ValueError(f"'{word}' not in vocabulary")

    poems = []
    hidden = None

    with torch.no_grad():
        _, hidden = feed_word(model, word2ix["<START>"], hidden, device)

        for i, head in enumerate(heads):
            line = [head]  # 每句首字强制为藏头字
            output, hidden = feed_word(model, word2ix[head], hidden, device)

            for _ in range(line_len - 1):
                next_idx, word = sample_word(
                    output, ix2word, word2ix,
                    temperature=temperature,
                    top_p=top_p
                )
                line.append(word)
                output, hidden = feed_word(model, next_idx, hidden, device)

            punctuation = "。" if i % 2 == 1 or i == len(heads) - 1 else "，"
            line.append(punctuation)
            if punctuation in word2ix:
                _, hidden = feed_word(model, word2ix[punctuation], hidden, device)

            poems.append("".join(line))

    return "\n".join(poems)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ix2word, word2ix = load_model(device)
    test_heads = sys.argv[1:] if len(sys.argv) > 1 else ["深度学习", "丰川祥子", "千早爱音", "吉野骏雄", "长崎素世","瓜皮帽就是区"]
    for heads in test_heads:
        poem = generate_acrostic(model, heads, ix2word, word2ix, device)
        print(f"藏头: {heads}")
        print(poem)
        print()
