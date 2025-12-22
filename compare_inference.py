#!/usr/bin/env python
import argparse
import sys
import pandas as pd

from transformer1000.inference import run_inference as run_transformer
from bertIDR1000.inference import run_model as run_bert

def read_sequences(args):
    if args.stdin:
        return [l.strip() for l in sys.stdin if l.strip()]

    if args.input.endswith((".fa", ".fasta")):
        seqs = []
        with open(args.input) as f:
            for line in f:
                if not line.startswith(">"):
                    seqs.append(line.strip())
        return seqs

    with open(args.input) as f:
        return [l.strip() for l in f if l.strip()]


def intersect_ranges(ranges1, ranges2):
    """
    求两个区间列表的交集
    """
    intersections = []

    for s1, e1 in ranges1:
        for s2, e2 in ranges2:
            s = max(s1, s2)
            e = min(e1, e2)
            if s < e:
                intersections.append((s, e))

    return intersections

def run_dual_inference(
    sequences,
    #bert_ckpt,
    #transformer_ckpt,
    device="cuda",
    min_length=4,
):
    """
    同时运行 BERT + Transformer
    """

    # ===== BERT =====
    df_bert = run_bert(
        sequences=sequences,
        #checkpoint=bert_ckpt,
        device=device,
        min_length=min_length,
    )

    df_bert = df_bert.rename(columns={
        "cedr_range": "bert_cedr_range",
        "cedr_len": "bert_cedr_len",
    })

    # ===== Transformer =====
    df_trans = run_transformer(
        sequences=sequences,
        #checkpoint=transformer_ckpt,
        device=device,
        min_length=min_length,
    )

    df_trans = df_trans.rename(columns={
        "cedr_range": "trans_cedr_range",
        "cedr_len": "trans_cedr_len",
    })

    # ===== Merge =====
    df = pd.DataFrame({
        "sequence": sequences,
        "bert_cedr_range": df_bert["bert_cedr_range"],
        "bert_cedr_len": df_bert["bert_cedr_len"],
        "trans_cedr_range": df_trans["trans_cedr_range"],
        "trans_cedr_len": df_trans["trans_cedr_len"],
    })

    # ===== Intersection =====
    df["overlap_range"] = df.apply(
        lambda r: intersect_ranges(
            r["bert_cedr_range"],
            r["trans_cedr_range"],
        ),
        axis=1,
    )

    df["overlap_len"] = df["overlap_range"].apply(
        lambda rs: sum(e - s for s, e in rs)
    )

    return df

def main():
    parser = argparse.ArgumentParser(
        description="Compare BERT and Transformer CEDR predictions"
    )

    parser.add_argument("--input", help="FASTA or text file")
    parser.add_argument("--stdin", action="store_true")
    #parser.add_argument("--bert-checkpoint", required=True)
    #parser.add_argument("--transformer-checkpoint", required=True)
    parser.add_argument("--output", default="compare_output.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-length", type=int, default=4)

    args = parser.parse_args()

    sequences = read_sequences(args)

    df = run_dual_inference(
        sequences=sequences,
        #bert_ckpt=args.bert_checkpoint,
        #transformer_ckpt=args.transformer_checkpoint,
        device=args.device,
        min_length=args.min_length,
    )

    df.to_csv(args.output, index=False)
    print(f"✅ Saved comparison to {args.output}")


if __name__ == "__main__":
    main()
