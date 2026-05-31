from collections import Counter
from pathlib import Path
import torch
from torch.utils.data import Dataset

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

DATA_ROOT = Path(__file__).resolve().parent.parent / "Data" / "NiuTrans"	# 数据集根目录
TRAIN_SRC_PATH = DATA_ROOT / "TM-training-set" / "chinese.txt"
TRAIN_TGT_PATH = DATA_ROOT / "TM-training-set" / "english.txt"
DEV_PATH = DATA_ROOT / "Dev-set" / "Niu.dev.txt"
TEST_PATH = DATA_ROOT / "Test-set" / "Niu.test.txt"
REFERENCE_PATH = DATA_ROOT / "Reference-for-evaluation" / "Niu.test.reference"
PROCESSED_PATH = DATA_ROOT / "processed.pt"	# 预处理后的缓存文件

MIN_FREQ = 2
MAX_VOCAB_SIZE = 30000
MAX_LEN = 80
TRAIN_LIMIT = None


class Vocabulary:	# 词表和 token-id 映射
    def __init__(self, token_to_idx):
        self.token_to_idx = token_to_idx
        self.idx_to_token = {idx: token for token, idx in token_to_idx.items()}
        self.pad_idx = token_to_idx[PAD_TOKEN]
        self.bos_idx = token_to_idx[BOS_TOKEN]
        self.eos_idx = token_to_idx[EOS_TOKEN]
        self.unk_idx = token_to_idx[UNK_TOKEN]

    def __len__(self):
        return len(self.token_to_idx)

    def encode(self, tokens, add_bos=False, add_eos=False, max_len=None):	# token 序列转 id 序列
        ids = []
        if add_bos:
            ids.append(self.bos_idx)

        for token in tokens:
            ids.append(self.token_to_idx.get(token, self.unk_idx))

        if add_eos:
            ids.append(self.eos_idx)

        if max_len is not None:
            ids = ids[:max_len]
            if add_eos and ids[-1] != self.eos_idx:
                ids[-1] = self.eos_idx
        return ids

    def decode(self, ids):	# id 序列还原为 token 序列
        tokens = []
        for idx in ids:
            token = self.idx_to_token.get(int(idx), UNK_TOKEN)
            if token in (BOS_TOKEN, PAD_TOKEN):
                continue
            if token == EOS_TOKEN:
                break
            tokens.append(token)
        return tokens

def tokenize(line):	# 数据已经分词，直接按空格切分
    return line.strip().split()


def read_lines(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def read_parallel_files(src_path, tgt_path, limit=None):	# 读取训练集平行语料
    src_lines = read_lines(src_path)
    tgt_lines = read_lines(tgt_path)
    pairs = []
    for src, tgt in zip(src_lines, tgt_lines):
        if src and tgt:
            pairs.append((tokenize(src), tokenize(tgt)))
        if limit is not None and len(pairs) >= limit:
            break
    return pairs


def read_interleaved_parallel_file(path):	# 读取交错排列的 dev/reference 文件
    """Dev/reference 文件为: 中文行、空行、英文行。"""
    lines = [line for line in read_lines(path) if line]
    pairs = []
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            pairs.append((tokenize(lines[i]), tokenize(lines[i + 1])))
    return pairs


def read_test_sources(path):	# 测试集只有源语言句子
    return [tokenize(line) for line in read_lines(path) if line]


def build_vocab(sentences, min_freq=2, max_size=30000):	# 根据训练集构造词表
    counter = Counter()
    for sent in sentences:
        counter.update(sent)

    token_to_idx = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if len(token_to_idx) >= max_size:
            break
        if token not in token_to_idx:
            token_to_idx[token] = len(token_to_idx)
    return Vocabulary(token_to_idx)


class IndexedTranslationDataset(Dataset):	# 保存已经数值化的翻译样本
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.pairs[idx]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)


def collate_fn(batch, src_pad_idx, tgt_pad_idx):	# batch 内动态 padding
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.nn.utils.rnn.pad_sequence(
        src_batch, batch_first=True, padding_value=src_pad_idx
    )
    tgt_batch = torch.nn.utils.rnn.pad_sequence(
        tgt_batch, batch_first=True, padding_value=tgt_pad_idx
    )
    return src_batch.long(), tgt_batch.long()


def encode_pairs(pairs, src_vocab, tgt_vocab, max_len):	# 平行句对数值化
    encoded = []
    for src_tokens, tgt_tokens in pairs:
        src_ids = src_vocab.encode(
            src_tokens, add_bos=True, add_eos=True, max_len=max_len
        )
        tgt_ids = tgt_vocab.encode(
            tgt_tokens, add_bos=True, add_eos=True, max_len=max_len
        )
        encoded.append((src_ids, tgt_ids))
    return encoded


def encode_sources(sources, src_vocab, max_len):	# 测试源句数值化
    return [
        src_vocab.encode(tokens, add_bos=True, add_eos=True, max_len=max_len)
        for tokens in sources
    ]


def preprocess_and_save(output_path=PROCESSED_PATH):	# 一次性预处理并保存
    train_pairs = read_parallel_files(TRAIN_SRC_PATH, TRAIN_TGT_PATH, limit=TRAIN_LIMIT)
    dev_pairs = read_interleaved_parallel_file(DEV_PATH)
    test_sources = read_test_sources(TEST_PATH)
    reference_pairs = read_interleaved_parallel_file(REFERENCE_PATH)

    src_vocab = build_vocab(
        [src for src, _ in train_pairs],
        min_freq=MIN_FREQ,
        max_size=MAX_VOCAB_SIZE
    )
    tgt_vocab = build_vocab(
        [tgt for _, tgt in train_pairs],
        min_freq=MIN_FREQ,
        max_size=MAX_VOCAB_SIZE
    )

    processed = {
        "src_vocab": src_vocab.token_to_idx,
        "tgt_vocab": tgt_vocab.token_to_idx,
        "train_pairs": encode_pairs(train_pairs, src_vocab, tgt_vocab, MAX_LEN),
        "dev_pairs": encode_pairs(dev_pairs, src_vocab, tgt_vocab, MAX_LEN),
        "test_sources": encode_sources(test_sources, src_vocab, MAX_LEN),
        "test_source_tokens": test_sources,
        "references": [tgt for _, tgt in reference_pairs],
        "max_len": MAX_LEN,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(processed, output_path)
    return processed


def load_processed_data(path=PROCESSED_PATH):	# Train/Evaluate 共用的数据入口
    path = Path(path)
    data = torch.load(path, weights_only=False)
    src_vocab = Vocabulary(data["src_vocab"])
    tgt_vocab = Vocabulary(data["tgt_vocab"])
    train_dataset = IndexedTranslationDataset(data["train_pairs"])
    dev_dataset = IndexedTranslationDataset(data["dev_pairs"])
    return data, train_dataset, dev_dataset, src_vocab, tgt_vocab


if __name__ == "__main__":
    data = preprocess_and_save(PROCESSED_PATH)
