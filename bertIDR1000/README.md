# CEDR  Prediction

This repository provides a BERT-based model for predicting
CEDR at residue resolution.
Usage
🔹 FASTA input
```bash
python inference.py \
  --input examples_test.fasta \
  --output result.csv \
  --device cpu 
```
🔹 txt input（one sequence per line）
```bash
python inference.py \
  --input sequences.txt \
  --device cpu 
```
🔹 standard input
```bash
echo "MKPGFSPRGGGFGGRGGFGDRGGRGGRGGF" | \
python inference.py --stdin --device cpu
```
