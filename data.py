"""
Datasets for the harness: char-level tiny-shakespeare and GPT-2-token FineWeb-Edu.

Both expose the same interface -- vocab_size / splits / get_batch / encode /
decode -- so everything downstream only has to pick one. Tiny-shakespeare is
held in memory; FineWeb-Edu is tokenized once into uint16 shards and read back
as memmaps, nanoGPT-style, because the corpus is far past what a colab VM will
hold.
"""
import json
import os
import urllib.request
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch as t

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = "data"
INPUT_PATH = os.path.join(DATA_DIR, "input.txt")

# FineWeb-Edu. The subsets are prefixes inside one repo: sample-10BT is 14
# parquet files, sample-100BT is 140, sample-350BT is 472, and "full" is every
# CC-MAIN dump (~1.3T tokens). Only the row groups actually needed are fetched,
# so the subset is a ceiling, not a download size
FINEWEB_REPO = "HuggingFaceFW/fineweb-edu"
FINEWEB_SUBSETS = ("sample-10BT", "sample-100BT", "sample-350BT", "full")
FINEWEB_SUBSET = FINEWEB_SUBSETS[0]
FINEWEB_TOKENS = 100_000_000  # default budget prepared on disk (~200 MB)

# One output shard per this many tokens (200 MB each). This is the resume
# granularity: an interrupted prep loses at most the shard it was writing
SHARD_TOKENS = 100_000_000
# The validation split is written once, before any training shard, and never
# grows: a corpus extended from 1B to 10B tokens keeps the same val set, so val
# losses stay comparable across budgets. Clipped for the toy budgets used in
# tests, where 5M tokens is more than the whole corpus
VAL_TOKENS = 5_000_000

# tiktoken's gpt2 encoding: 50257 ids, which is exactly gpt2-small's embedding.
# nanoGPT pads this to 50304 for the matmul shape; kept exact here because the
# point of the gpt2 preset is to be the published configuration.
GPT2_VOCAB = 50257
GPT2_EOT = 50256  # <|endoftext|>, the document separator


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

    name = "shakespeare"

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


def human_tokens(n: int) -> str:
    """
    1_500_000 -> '1.50M'; used in every log line
    """
    for scale, suffix in ((1_000_000_000, "B"), (1_000_000, "M"),
                          (1_000, "K")):
        if n >= scale:
            v = n / scale
            return f"{v:.0f}{suffix}" if v == int(v) else f"{v:.2f}{suffix}"
    return str(n)


# ============================================================================
# FineWeb-Edu preparation
# ============================================================================

def corpus_dir(subset: str = FINEWEB_SUBSET) -> str:
    """
    Where a subset is tokenized to. The budget is deliberately not part of the
    name: a corpus grows in place, so asking for 10B after preparing 1B extends
    the shards instead of re-downloading the first 1B into a second directory
    """
    return os.path.join(DATA_DIR, f"fineweb_edu_{subset}")


def _subset_prefix(subset: str) -> str:
    if subset == "full":
        return "data/"
    return f"sample/{subset.split('-', 1)[1]}/"


def fineweb_files(subset: str) -> List[str]:
    """
    The subset's parquet files, in repo order
    """
    from huggingface_hub import HfApi

    prefix = _subset_prefix(subset)
    files = sorted(f for f in HfApi().list_repo_files(FINEWEB_REPO,
                                                      repo_type="dataset")
                   if f.startswith(prefix) and f.endswith(".parquet"))
    if not files:
        raise RuntimeError(f"no parquet files under {prefix!r} in "
                           f"{FINEWEB_REPO}; is --data-subset right?")
    return files


def _iter_row_groups(files: List[str], start_file: int, start_group: int):
    """
    Yield (file index, group index, [texts]) from a checkpoint onwards.

    Parquet row groups rather than `datasets`' streaming iterator: a
    (file, group) pair is a position the reader can jump straight back to with
    one HTTP range request (~0.7 s on a 2 GB file), which is what makes a
    10B-token preparation resumable instead of merely restartable. Streaming
    would have to re-read -- and re-download -- everything before the cut.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    for fi in range(start_file, len(files)):
        with fs.open(f"datasets/{FINEWEB_REPO}/{files[fi]}", "rb") as f:
            pf = pq.ParquetFile(f)
            first = start_group if fi == start_file else 0
            for gi in range(first, pf.num_row_groups):
                table = pf.read_row_group(gi, columns=["text"])
                yield fi, gi, table.column("text").to_pylist()


def _fresh_meta(subset: str) -> dict:
    return {"repo": FINEWEB_REPO, "subset": subset, "encoding": "gpt2",
            "vocab_size": GPT2_VOCAB, "dtype": "uint16",
            "shard_tokens": SHARD_TOKENS,
            "val": None,            # {"file": ..., "tokens": ...}
            "train_shards": [],     # same, in order
            "next_file": 0, "next_group": 0}


def read_meta(out_dir: str, subset: str = FINEWEB_SUBSET) -> dict:
    path = os.path.join(out_dir, "meta.json")
    if not os.path.exists(path):
        return _fresh_meta(subset)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def prepared_tokens(meta: dict) -> int:
    return sum(s["tokens"] for s in meta["train_shards"])


def prepare_fineweb_edu(target_tokens: int = FINEWEB_TOKENS,
                        subset: str = FINEWEB_SUBSET,
                        out_dir: Optional[str] = None,
                        val_tokens: Optional[int] = None) -> str:
    """
    Tokenize FineWeb-Edu until the corpus holds target_tokens training tokens.

    Documents are read row group by row group, tokenized with tiktoken's gpt2
    encoding and terminated by <|endoftext|>. The validation split is written
    first and only once; training tokens then accumulate in SHARD_TOKENS-sized
    shards.

    Resumable and extensible, both through the same mechanism: `meta.json`
    records the shards that are complete and the (file, row group) to read
    next, and is rewritten after every completed shard. So an interrupted 10B
    run continues where it stopped, losing at most the partial shard and the
    row group that straddled it, and a later call with a larger budget appends
    shards to the same directory rather than starting over. A call whose budget
    is already on disk does nothing, which makes it safe at the top of a run.
    """
    out_dir = out_dir or corpus_dir(subset)
    meta = read_meta(out_dir, subset)
    if meta["subset"] != subset:
        raise RuntimeError(f"{out_dir} holds {meta['subset']}, not {subset}")
    have = prepared_tokens(meta)
    if have >= target_tokens:
        return out_dir

    import tiktoken
    from tqdm.auto import tqdm

    os.makedirs(out_dir, exist_ok=True)
    # a shard not listed in meta is the one an interrupted run was writing
    listed = {s["file"] for s in meta["train_shards"]}
    if meta["val"]:
        listed.add(meta["val"]["file"])
    for name in os.listdir(out_dir):
        if name.endswith(".bin") and name not in listed:
            print(f"  dropping partial shard {name}")
            os.remove(os.path.join(out_dir, name))

    enc = tiktoken.get_encoding("gpt2")
    files = fineweb_files(subset)
    groups = _iter_row_groups(files, meta["next_file"], meta["next_group"])
    state = {"buf": np.empty(0, dtype=np.uint16)}

    def save() -> None:
        with open(os.path.join(out_dir, "meta.json"), "w",
                  encoding="utf-8") as f:
            json.dump(meta, f, indent=1)

    def refill() -> None:
        try:
            fi, gi, texts = next(groups)
        except StopIteration:
            raise RuntimeError(
                f"{subset} exhausted after {prepared_tokens(meta):,d} training "
                f"tokens, wanted {target_tokens:,d} -- prepare a larger "
                f"--data-subset (one of {FINEWEB_SUBSETS})") from None
        # a batch of documents per call: tiktoken parallelises inside
        # encode_ordinary_batch, one document at a time does not
        chunks = [np.asarray(ids + [GPT2_EOT], dtype=np.uint16)
                  for ids in enc.encode_ordinary_batch(texts)]
        state["buf"] = np.concatenate([state["buf"], *chunks])
        # the position to resume from is the group *after* the one just read;
        # the leftover in buf is dropped on resume, which costs at most one row
        # group of documents and never duplicates any
        meta["next_file"], meta["next_group"] = fi, gi + 1

    def write_file(name: str, quota: int, bar) -> int:
        done = 0
        with open(os.path.join(out_dir, name), "wb") as f:
            while done < quota:
                if len(state["buf"]) == 0:
                    refill()
                take = min(len(state["buf"]), quota - done)
                state["buf"][:take].tofile(f)
                state["buf"] = state["buf"][take:]
                done += take
                bar.update(take)
        return done

    if meta["val"] is None and val_tokens is None:
        # clipped for the toy budgets used in tests, where the fixed 5M would
        # be larger than the whole corpus
        val_tokens = min(VAL_TOKENS, max(1024, target_tokens // 20))
    todo = target_tokens - have + (0 if meta["val"] else val_tokens)
    print(f"preparing {FINEWEB_REPO}/{subset} in {out_dir}: "
          f"{human_tokens(have)} -> {human_tokens(target_tokens)} train tokens")
    bar = tqdm(total=todo, unit="tok", unit_scale=True, desc="tokenizing")
    try:
        if meta["val"] is None:
            n = write_file("val.bin", val_tokens, bar)
            meta["val"] = {"file": "val.bin", "tokens": n}
            save()
        while have < target_tokens:
            name = f"train_{len(meta['train_shards']):05d}.bin"
            n = write_file(name, min(SHARD_TOKENS, target_tokens - have), bar)
            meta["train_shards"].append({"file": name, "tokens": n})
            have += n
            save()
    finally:
        bar.close()
    return out_dir


# ============================================================================
# reading a tokenized corpus
# ============================================================================

class ShardedTokens:
    """
    uint16 shards addressed as one token array.

    Sampling never crosses a shard boundary -- with 100M-token shards that
    forfeits one sequence per shard, and it keeps every window a single memmap
    slice instead of a concatenation.
    """

    def __init__(self, paths: List[str], limit: Optional[int] = None):
        self.parts: List[np.memmap] = []
        self.lens: List[int] = []
        total = 0
        for p in paths:
            if limit is not None and total >= limit:
                break
            part = np.memmap(p, dtype=np.uint16, mode="r")
            n = len(part) if limit is None else min(len(part), limit - total)
            self.parts.append(part)
            self.lens.append(int(n))
            total += n
        self._cum: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return int(sum(self.lens))

    def _cumulative(self, block_size: int) -> np.ndarray:
        """
        Cumulative count of legal start offsets, so a uniform draw over the
        corpus is one searchsorted rather than a per-shard choice
        """
        if block_size not in self._cum:
            usable = [max(n - block_size - 1, 0) for n in self.lens]
            self._cum[block_size] = np.cumsum([0] + usable)
        return self._cum[block_size]

    def n_starts(self, block_size: int) -> int:
        return int(self._cumulative(block_size)[-1])

    def window(self, start: int, block_size: int) -> np.ndarray:
        """
        block_size + 1 tokens from the start-th legal offset: X and its shifted Y
        """
        cum = self._cumulative(block_size)
        j = int(np.searchsorted(cum, start, side="right")) - 1
        off = start - int(cum[j])
        return self.parts[j][off:off + block_size + 1]


class TokenDataset:
    """
    A pre-tokenized corpus, read as memmapped uint16 shards.

    Batches are sampled the way CharDataset samples them -- uniform random
    offsets, no epoch bookkeeping -- so the two are interchangeable everywhere
    in the harness.
    """

    name = "fineweb-edu"

    def __init__(self, data_dir: str, tokens: Optional[int] = None):
        self.data_dir = data_dir
        self.meta = read_meta(data_dir)
        self.vocab_size = self.meta["vocab_size"]
        shards = [os.path.join(data_dir, s["file"])
                  for s in self.meta["train_shards"]]
        self.splits = {
            "train": ShardedTokens(shards, limit=tokens),
            "val": ShardedTokens([os.path.join(data_dir,
                                               self.meta["val"]["file"])]),
        }
        self._enc = None

    @property
    def enc(self):
        if self._enc is None:
            import tiktoken
            self._enc = tiktoken.get_encoding(self.meta["encoding"])
        return self._enc

    def encode(self, s: str) -> t.Tensor:
        return t.tensor(self.enc.encode_ordinary(s), dtype=t.long)

    def decode(self, ids) -> str:
        return self.enc.decode([int(i) for i in ids])

    def get_batch(self, split: str, batch_size: int, block_size: int,
                  device: t.device, generator=None) -> Tuple[t.Tensor, t.Tensor]:
        data = self.splits[split]
        n = data.n_starts(block_size)
        assert n > 0, (f"the {split} split holds {len(data):,d} tokens, "
                       f"shorter than block_size + 1 = {block_size + 1}")
        ix = t.randint(n, (batch_size,), generator=generator).tolist()
        # stacked in numpy and cast once: torch has no uint16 arithmetic, and
        # slicing the memmap per sequence in torch would fault a page at a time
        w = np.stack([data.window(i, block_size) for i in ix]).astype(np.int64)
        return (t.from_numpy(w[:, :-1]).to(device, non_blocking=True),
                t.from_numpy(w[:, 1:]).to(device, non_blocking=True))


DATASETS = ("shakespeare", "fineweb-edu")

# what the harness annotates a corpus with: the two classes above share an
# interface but no base class
Dataset = Union[CharDataset, TokenDataset]


def get_dataset(name: str = "shakespeare",
                tokens: int = FINEWEB_TOKENS,
                subset: str = FINEWEB_SUBSET,
                val_frac: Optional[float] = None) -> Dataset:
    """
    Build one of DATASETS. FineWeb-Edu is prepared (or extended) on first use.
    """
    if name == "shakespeare":
        return CharDataset(**({"val_frac": val_frac} if val_frac else {}))
    if name == "fineweb-edu":
        out = prepare_fineweb_edu(tokens, subset)
        return TokenDataset(out, tokens)
    raise ValueError(f"unknown dataset {name!r}, expected one of {DATASETS}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="prepare / inspect a dataset. Preparation is resumable and "
                    "extensible: re-running with a larger --tokens appends "
                    "shards to the same directory")
    ap.add_argument("--dataset", default="shakespeare", choices=DATASETS)
    ap.add_argument("--tokens", type=int, default=FINEWEB_TOKENS,
                    help="fineweb-edu: gpt2 training tokens to hold on disk")
    ap.add_argument("--subset", default=FINEWEB_SUBSET,
                    choices=list(FINEWEB_SUBSETS))
    args = ap.parse_args()

    ds = get_dataset(args.dataset, args.tokens, args.subset)
    print(f"{ds.name}: vocab_size={ds.vocab_size}")
    print({k: f"{len(v):,d} tokens" for k, v in ds.splits.items()})
    X, Y = ds.get_batch("train", 4, 32, t.device("cpu"))
    print(X.shape, Y.shape)
    print(repr(ds.decode(X[0])))
