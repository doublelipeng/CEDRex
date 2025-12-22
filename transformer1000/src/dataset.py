import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class Vocab:
    def __init__(self, tokens=None):
        self.idx_to_token = list()
        self.token_to_idx = dict()
        if tokens is not None:
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    @classmethod
    def build(cls, text):
        uniq_tokens = ['<pad>', 'M', 'K', 'S', 'A', 'R', 'G', 'W', 'D', 'Q', 
                       'F', 'V', 'E', 'P', 'H', 'I', 'Y', 'L', 'N', 'T', 'C']
        return cls(uniq_tokens)

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.token_to_idx['<pad>'])

    def convert_tokens_to_ids(self, tokens):
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, indices):
        return [self.idx_to_token[index] for index in indices]


class TransformerDataset(Dataset):
    """存储已编码好的序列（List of (ids, valid_length)）"""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate_fn(examples, vocab, max_len=1000):
    """padding 同 batch 内最长序列"""
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    inputs = [torch.tensor(ex[0]) for ex in examples]
    valid_length = torch.tensor([ex[1] for ex in examples])

    inputs = pad_sequence(inputs, batch_first=True, padding_value=vocab["<pad>"])
    return inputs, lengths, valid_length
