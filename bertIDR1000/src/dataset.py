import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class TransformerDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate_fn(examples,pad_id: int, inputmax_length=1000):
    """
    wfj:该函数表示对于batch_size中的每一个元素做以下一下的操作，通常用来进行数据的标准化工作
    """
    inputs = [torch.tensor(ex[:inputmax_length - 1]) for ex in examples]
    lengths = torch.tensor([len(inp) for inp in inputs])
    # 对batch内的样本进行padding，使其具有相同长度
    inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_id)
    
    return inputs, lengths