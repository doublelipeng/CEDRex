# LARKS-Transformer

**LARKS-Transformer** — a transformer-based predictor for LARKS (Low-complexity Aromatic-rich Kinked Segments) in protein sequences.

This repository contains the model code, training & evaluation scripts, and example notebooks for the LARKS Transformer described in our work. The model predicts whether each residue in a protein sequence belongs to a LARKS segment.

---

## ⚡ Highlights

- Input: full protein amino-acid sequence (or IDR subsequences when preferred).
- Architecture: Encoder-Decoder Transformer:
  - Encoder: 3 layers, 3 heads (PreNorm).
  - Decoder: 3 layers, 3 heads; decoder attends to encoder output (memory).
  - Embeddings: uses frozen ESM-2 (`esm2_t6_8M_UR50D`) pretrained embeddings (dim = 320).
- Tokens: 20 canonical amino acids + special tokens `{<bos>, <eos>, <pad>}`.
- Training tricks: teacher forcing, scheduled sampling (for fine-tuning), greedy decoding at inference with a strategy to avoid premature EOS.
- Input length: model trained up to sequence length 1000; longer proteins are split/truncated.
- Performance: AUC (on leave-one-cluster-out splits) after scheduled sampling: **~71.5%–80.6%**.

---

## 🔧 Installation

Requires Python 3.8+ and PyTorch (CUDA recommended for training). Example minimal environment setup:

```bash
# optional: create env and install
conda create -n larks python=3.8 -y
conda activate larks

# install pytorch (choose appropriate CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# other dependencies
pip install numpy pandas scikit-learn
```
⚠️ Caveats & Recommendations

Amino acids: Model trained on standard 20 amino acids + 3 tokens (bos/eos/pad). It cannot predict sequences with nonstandard residues or rare amino acids outside the 20 canonical ones.
Input strategy: Using whole-protein input is convenient and lets structured (non-IDR) regions serve as negative examples, but if you have reliable IDR annotations, try both strategies (whole-protein vs IDR subsequence) — results differ subtly.
Small data: With limited labeled LARKS instances, regularization and freezing pretrained embeddings help. You may consider data augmentation (motif shuffling, reverse complement of peptide sequences? — careful with biological plausibility), or weakly-supervised / semi-supervised approaches.
🔬 Example inference (python)
```python
from src.model import LarksTransformer
from src.dataset import TransformerDataset, Vocab, DataLoader
import torch, numpy as np
vocab=Vocab(sents)
# load model
model = LarksTransformer(...)
model.load_state_dict(torch.load("checkpoints/best.pt"))
model.eval()

test_data=[vocab.convert_tokens_to_ids(sentence) for sentence in sents]
test_dataset = TransformerDataset(test_data)
test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

with torch.no_grad():
    for batch in test_data_loader:
        inputs, lengths,valid_length = [x for x in batch]
        l = lengths.cpu().item() 
        pred = model.evaluate(inputs,src_lengths=lengths,min_length=l+2)
print(pred.shape) 

```
📑 Citation

If you use this repository / model in your work, please cite:
Lipeng Li, HKLab. LARKS-Transformer: Transformer-based predictor for LARKS. GitHub repository. (Year).
You may also cite the core papers related to ESM-2 if you use its embeddings.
