# CEDRex

**CEDRex** predicts **Cross-β-interaction Enriched Dynamic Regions (CEDRs)** in proteins using complementary Transformer-based models.

---

## Overview

CEDRex integrates two models:

1. **Encoder–Decoder Transformer** – trained on **whole protein sequences**  
2. **BERT-style Transformer** – trained on **intrinsically disordered regions (IDRs)**  

Users can run either model individually or both together. Overlapping predictions can be used for consensus analysis.

---

## Input
CEDRex supports both whole-protein and IDR-based predictions under the following constraints:

- The maximum supported input length is **1000 amino acids** for both full-length protein sequences and IDR segments. Longer sequences must be truncated prior to inference.
- IDRs are defined using **MobiDB-lite**, followed by structural filtering using **AlphaFold2**, where regions predicted to form helices with confidence greater than **70%** are excluded.
- Only sequences composed of the **20 standard amino acids** are supported. Sequences containing rare, ambiguous, or non-canonical residues can not be predicted.

**References:**  
- Necci M et al., MobiDB-lite 3.0: fast consensus annotation of intrinsic disorder flavors in proteins. Bioinformatics. 2020  
- Fleming J. et al. AlphaFold Protein Structure Database and 3D-Beacons: New Data and Capabilities. Journal of Molecular Biology, 2025
- Jumper, J et al. Highly accurate protein structure prediction with AlphaFold. Nature 2021 

---

## Installation & Dependencies

- Python ≥ 3.8  
- PyTorch 2.0.1+cu117  choose appropriate version base on your computer
- Biopython  

```bash

git clone https://github.com/doublelipeng/CEDRex.git
cd CEDRex
```

Usage
Predict using both models:
🔹 standard input
```
echo "MKPGFSPRGGGFGGRGGFGDRGGRGGRGGF" | python compare_inference.py --stdin --device cpu
```

🔹 FASTA input
```bash
python inference.py \
  --input transfomer1000/examples_test.fasta \
  --output result.csv \
  --device cpu 
```

🔹 txt input（one sequence per line）
```bash
python inference.py \
  --input transfomer1000/sequences.txt
```

Predict with only BERT (IDR regions): please follow the readme file in bertIDR1000

Predict with only Transformer (whole protein or IDRs): please follow the readme file in transfomer1000

---
Output CSV columns:

Ranges are **0-based**, with the **start position inclusive** and the **end position exclusive**.

1.bert_cedr_range – predicted CEDR ranges by BERT

2.trans_cedr_range – predicted CEDR ranges by Transformer

3.overlap_range – intersection of the two predictions

---

Citation

If you use CEDRex, please cite: HKlab, Dynamic Secondary Structure Orchestrates Regulated Phase Separation and Synaptic Transmission
