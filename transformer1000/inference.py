import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
try:
    from .src.model import Transformer
    from .src.dataset import Vocab, TransformerDataset, collate_fn
except ImportError:
    from src.model import Transformer
    from src.dataset import Vocab, TransformerDataset, collate_fn


# ======================
# Utils
# ======================
def find_consecutive_ranges(data, target=4, min_length=1):
    ranges = []
    start = None
    for i, value in enumerate(data):
        if value == target:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - start >= min_length:
                    ranges.append((start, i))
                start = None
    if start is not None and len(data) - start >= min_length:
        ranges.append((start, len(data)))
    return ranges


def strip_module_prefix(state_dict):
    if list(state_dict.keys())[0].startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict

# ======================
# Inference API
# ======================
from pathlib import Path
@torch.no_grad()
def run_inference(
    sequences,
    checkpoint='Transformer_maxlen1000_model_whole_seq_state_dict.pt',
    device="cuda",
    max_len=1000,
    min_length=1,
):
    """
    Args:
        sequences (List[str])
        checkpoint (str): model path
        device (str)
        max_len (int)
        min_length (int)

    Returns:
        pd.DataFrame
    """
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

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # ===== Build vocab =====
    vocab = Vocab.build(sequences)

    seq_lens = [len(s) for s in sequences]
    test_data = [
        (vocab.convert_tokens_to_ids(s), l)
        for s, l in zip(sequences, seq_lens)
    ]

    dataset = TransformerDataset(test_data)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=partial(collate_fn, vocab=vocab, max_len=max_len),
    )

    # ===== Load model =====
    model = Transformer(vocab_size=21, hidden_dim=320, num_class=5)
    state_dict = torch.load(checkpoint_path, map_location=device)
    state_dict = strip_module_prefix(state_dict)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    tag_vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "uncedr": 3,
        "cedr": 4,
    }

    predictions = []

    # ===== Predict =====
    for batch in tqdm(loader, desc="Predicting"):
        inputs, lengths, valid_length = [x.to(device) for x in batch]
        l = lengths.item()

        pred = model.evaluate_stop_on_eos(
            inputs,
            src_lengths=lengths,
            tag_vocab=tag_vocab,
            min_length=l + 2,
        )

        pred = (
            pred.view(-1)[1:-1]
            .detach()
            .cpu()
            .numpy()[:l]
        )

        predictions.append(pred)

    # ===== Build output =====
    df = pd.DataFrame({
        "sequence": sequences,
        "prediction": predictions,
    })

    df["cedr_range"] = df["prediction"].apply(
        lambda x: find_consecutive_ranges(x, target=4, min_length=min_length)
    )
    df["cedr_len"] = df["cedr_range"].apply(
        lambda r: sum(e - s for s, e in r)
    )

    return df

#!/usr/bin/env python
import argparse
import sys

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

def read_sequences(args):
    if args.stdin:
        sequences=[line.strip() for line in sys.stdin if line.strip()]
        return sequences,[f"seq{i}" for i in range(len(sequences))]

    elif args.input.endswith(".fasta") or args.input.endswith(".fa"):
        names, sequences = read_fasta(args.input)
        return sequences,names

    else:
        with open(args.input) as f:
            sequences=[line.strip() for line in f if line.strip()]
            return sequences,[f"seq{i}" for i in range(len(sequences))]


def main():
    parser = argparse.ArgumentParser(
        description="CEDR prediction using Transformer1000"
    )

    parser.add_argument("--input", help="FASTA or text file")
    parser.add_argument("--stdin", action="store_true", help="Read sequences from stdin")
    #parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="output.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-len", type=int, default=1000)
    parser.add_argument("--min-length", type=int, default=1)

    args = parser.parse_args()

    sequences,names = read_sequences(args)

    df = run_inference(
        sequences=sequences,
        #checkpoint=args.checkpoint,
        device=args.device,
        max_len=args.max_len,
        min_length=args.min_length,
    )
    if names is not None:
        df.insert(0, "id", names)

    df.to_csv(args.output, index=False)
    print(f"✅ Saved results to {args.output}")


if __name__ == "__main__":
    main()
