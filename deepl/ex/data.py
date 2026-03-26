import torch

from ex.tokenizer import SimpleTokenizer


class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list[str], labels: list[int], max_vocab: int, max_seq_length: int):
        self.text = texts
        self.labels = labels
        self.max_seq_length = max_seq_length
        self.tokenizer = SimpleTokenizer(
            corpus=" ".join(texts), max_number_tokens=max_vocab
        )

    def __len__(self):
        return len(self.text)

    def label_to_one_hot(self, label: int) -> torch.Tensor:
        if label == 0:
            return torch.tensor([1., 0.])
        return torch.tensor([0., 1.])

    def __getitem__(self, item) -> tuple[torch.tensor, int]:
        return torch.tensor(
            self.tokenizer.tokenize(self.text[item], length=self.max_seq_length)
        ), self.label_to_one_hot(self.labels[item])
