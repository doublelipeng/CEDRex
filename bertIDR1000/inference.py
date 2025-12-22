#!/usr/bin/env python
import argparse
import sys
import os
import torch
import pandas as pd
from collections import defaultdict
from functools import partial
from torch.utils.data import DataLoader

try:
    # 作为 package 被 import / python -m
    from .src.model import BERTForTokenClassification
    from .src.dataset import TransformerDataset, collate_fn
    from .src.vocab import Vocab
except ImportError:
    # 作为脚本直接运行
    from src.model import BERTForTokenClassification
    from src.dataset import TransformerDataset, collate_fn
    from src.vocab import Vocab


# =========================
# Utils
# =========================
def read_fasta(path):
    seqs = []
    names = []
    seq = []
    name = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq:
                    seqs.append("".join(seq))
                    names.append(name)
                    seq = []
                name = line[1:]
            else:
                seq.append(line)
        if seq:
            seqs.append("".join(seq))
            names.append(name)

    return names, seqs


def find_consecutive_ranges(data, target=24, min_length=1):
    ranges = []
    start = None
    for i, v in enumerate(data):
        if v == target:
            if start is None:
                start = i
        else:
            if start is not None and i - start >= min_length:
                ranges.append((start, i))
            start = None
    if start is not None and len(data) - start >= min_length:
        ranges.append((start, len(data)))
    return ranges

from pathlib import Path
import torch
@torch.no_grad()
def run_model(sequences, checkpoint='bert_maxlen1000_model_RoPE_state_dict.pt', device="cuda:0", batch_size=1, min_length=2):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    this_dir = Path(__file__).resolve().parent
    # 支持两种情况：
    # 1) 用户传绝对路径
    # 2) 用户只传文件名
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = this_dir / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )
    vocab = Vocab.build()
    pad_id = vocab["<pad>"]

    model = BERTForTokenClassification()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    tokenized = [vocab.convert_tokens_to_ids(['<bos>'] + list(s))
    for s in sequences]
    dataset = TransformerDataset(tokenized)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, pad_id=pad_id),
    )

    results = defaultdict(list)

    for batch in loader:
        inputs, lengths = [x.to(device) for x in batch]
        logits = model(inputs)
        preds = torch.argmax(logits, dim=-1)

        for i in range(inputs.size(0)):
            pred = preds[i].cpu().numpy()[1:]
            seq = vocab.convert_ids_to_tokens(inputs[i].cpu().numpy())[1:]

            ranges = find_consecutive_ranges(pred, target=24, min_length=min_length)

            results["sequence"].append("".join(seq))
            results["prediction"].append(pred)
            results["cedr_range"].append(ranges)
            results["cedr_len"].append(sum(e - s for s, e in ranges))

        del inputs, logits, preds
        torch.cuda.empty_cache()

    return pd.DataFrame(results)


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="CEDR prediction using BERT-IDR model"
    )

    parser.add_argument("--input", type=str, help="Input FASTA or TXT file")
    parser.add_argument("--stdin", action="store_true", help="Read sequence from stdin")
    #parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--output", type=str, default="prediction.csv", help="Output CSV")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--min-length", type=int, default=1)

    args = parser.parse_args()

    if args.stdin:
        sequences = [line.strip() for line in sys.stdin if line.strip()]
        names = [f"seq{i}" for i in range(len(sequences))]
    else:
        if args.input is None:
            parser.error("Either --input or --stdin must be provided")
        if args.input.endswith(".fasta") or args.input.endswith(".fa"):
            names, sequences = read_fasta(args.input)
        else:
            with open(args.input,'r') as f:
                sequences = [line.strip() for line in f]
            names = [f"seq{i}" for i in range(len(sequences))]

    df = run_model(
        sequences=sequences,
        #checkpoint=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        min_length=args.min_length,
    )

    df.insert(0, "id", names)
    df.to_csv(args.output, index=False)

    print(f"[✓] Prediction finished → {args.output}")


if __name__ == "__main__":
    main()
