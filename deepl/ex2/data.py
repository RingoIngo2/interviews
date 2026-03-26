import torch
from datasets import load_dataset
from ex2.tokenizer import Tokenizer


class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list[str], labels: list[int], max_number_tokens: int, max_length: int = 40):
        self.max_length = max_length
        self.texts = texts
        self.labels = labels
        self.tokenizer = Tokenizer(" ".join(texts), max_number_tokens)

    def __len__(self):
        return len(self.texts)

    @staticmethod
    def _label_to_one_hot(label: int) -> list[int]:
        if label:
            return torch.Tensor([1.0, 0.0])
        return torch.Tensor([0.0, 1.0])

    def __getitem__(self, item):
        return torch.tensor(
            self.tokenizer.tokenize(self.texts[item], max_length=self.max_length)
        ), self._label_to_one_hot(self.labels[item])


if __name__ == "__main__":
    data = load_dataset("imdb")["train"]
    dataset = IMDBDataset(list(data["text"]), list(data["label"]), max_number_tokens=10)
    print(dataset[4])
    print(len(dataset))
