import os
import urllib.request
from typing import Tuple

import torch as t

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = "data"
INPUT_PATH = os.path.join(DATA_DIR, "input.txt")


def download() -> str:
    """
    Fetch tiny-shakespeare once, return the raw text
    """
    if not os.path.exists(INPUT_PATH):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"downloading tiny-shakespeare -> {INPUT_PATH}")
        urllib.request.urlretrieve(URL, INPUT_PATH)
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        return f.read()


class CharDataset:
    """
    Char-level tiny-shakespeare, 90/10 split, held in memory as uint16
    """

    def __init__(self, val_frac: float = 0.1):
        text = download()
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}

        data = t.tensor([self.stoi[c] for c in text], dtype=t.uint16)
        n = int(len(data) * (1 - val_frac))
        self.splits = {"train": data[:n], "val": data[n:]}

    def encode(self, s: str) -> t.Tensor:
        return t.tensor([self.stoi[c] for c in s], dtype=t.long)

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def get_batch(self, split: str, batch_size: int, block_size: int,
                  device: t.device, generator=None) -> Tuple[t.Tensor, t.Tensor]:
        data = self.splits[split]
        ix = t.randint(len(data) - block_size - 1, (batch_size,),
                       generator=generator)
        X = t.stack([data[i:i + block_size] for i in ix]).long()
        Y = t.stack([data[i + 1:i + 1 + block_size] for i in ix]).long()
        return X.to(device, non_blocking=True), Y.to(device, non_blocking=True)


if __name__ == "__main__":
    ds = CharDataset()
    print(f"vocab_size={ds.vocab_size}")
    print({k: len(v) for k, v in ds.splits.items()})
    X, Y = ds.get_batch("train", 4, 32, t.device("cpu"))
    print(X.shape, Y.shape)
    print(repr(ds.decode(X[0])))
