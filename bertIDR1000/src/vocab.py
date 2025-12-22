class Vocab:
    def __init__(self, tokens=None):
        self.idx_to_token = []
        self.token_to_idx = {}

        if tokens is not None:
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    @classmethod
    def build(cls):
        uniq_tokens = [
            '<pad>', 'M', 'K', 'S', 'A', 'R', 'G', 'W', 'D', 'Q',
            'F', 'V', 'E', 'P', 'H', 'I', 'Y', 'L', 'N', 'T', 'C',
            '<bos>', '<eos>', 'uncedr', 'cedr'
        ]
        return cls(uniq_tokens)

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        return self.token_to_idx[token]

    def convert_tokens_to_ids(self, tokens):
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, indices):
        return [self.idx_to_token[i] for i in indices]
