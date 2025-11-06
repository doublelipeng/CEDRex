import torch, pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.model import LarksTransformer
from src.dataset import Vocab, TransformerDataset, collate_fn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======= Load data =======
concat_df = pd.read_csv("mobiDB/input_sequences.csv")  # 例如包含 UniProt_ID 和 Sequence
sents = list(concat_df["Sequence"])
seq_len = list(concat_df["Sequence"].apply(len))

vocab = Vocab.build(sents)
test_data = [(vocab.convert_tokens_to_ids(s), l) for s, l in zip(sents, seq_len)]

test_dataset = TransformerDataset(test_data)
test_loader = DataLoader(
    test_dataset, batch_size=1,
    collate_fn=lambda x: collate_fn(x, vocab, max_len=1000),
    shuffle=False
)

# ======= Load model =======
tag_vocab = {'<pad>': 0, '<bos>': 1, '<eos>': 2, 'unlark': 3, 'lark': 4}
model = torch.load("checkpoints/best.pt", map_location=device)
if isinstance(model, torch.nn.DataParallel):
    model = model.module
model = model.to(device)
model.eval()

# ======= Predict =======
predictions, all_inputs = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        inputs, lengths, valid_length = [x.to(device) for x in batch]
        l = lengths.cpu()[0].item()
        pred = model.evaluate(inputs, src_lengths=lengths, tag_vocab=tag_vocab, min_length=l+2)
        pred = pred.view(-1)[1:-1].detach().cpu().numpy()[:l]
        predictions.append(pred.copy())
        all_inputs.append(inputs.cpu().numpy()[-valid_length:])

# ======= Save =======
with open("results/larks_predictions.pkl", "wb") as f:
    pickle.dump({'prediction': predictions, 'inputs': all_inputs}, f)

print(f"✅ Saved predictions for {len(predictions)} sequences.")
