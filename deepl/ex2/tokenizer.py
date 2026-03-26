from functools import cached_property
from collections import Counter


class Tokenizer:
    EOS: str = "<EOS>"
    SOS: str = "<SOS>"
    UNKNOWN: str = "<UNKNOWN>"
    PADDING: str = "<PADDING>"

    def __init__(self, corpus: str, max_number_tokens: int = 100):
        self.max_number_tokens = max_number_tokens
        self.id2token = self._fit(corpus)

    @cached_property
    def token2id(self) -> dict[str, int]:
        return {t: i for i, t in self.id2token.items()}

    def token_id(self, token: str) -> int:
        return self.token2id.get(token, 3)

    def padding_idx(self) -> int:
        return self.token_id(self.PADDING)

    def _fit(self, corpus: str) -> dict[int, str]:
        id2token = {0: self.PADDING, 1: self.SOS, 2: self.EOS, 3: self.UNKNOWN}
        tokens = Counter(corpus.lower().split())
        most_common = tokens.most_common(self.max_number_tokens - 4)
        id2token.update({i + 4: t for i, (t, _) in enumerate(most_common)})
        return id2token

    def tokenize(self, s: str, max_length: int) -> list[int]:
        tokens_s = s.lower().split()  # bcz SOS token
        cropped_tokens_s = tokens_s[: min(len(tokens_s), max_length - 2)]
        n_padding = max_length - len(tokens_s) + 2
        cropped_tokens_s = (
            [self.SOS] + cropped_tokens_s + [self.EOS] + ([self.PADDING] * n_padding)
        )
        return [self.token_id(t) for t in cropped_tokens_s]

    def de_tokenize(self, tokens: list[int]) -> str:
        return "".join([self.id2token[i] for i in tokens])


if __name__ == "__main__":
    corpus = "hi, how are you today MY DEAR"
    s = "dear I am feeling good TODAY"
    T = Tokenizer(corpus)
    token_ids_s = T.tokenize(s, 10)
    print(token_ids_s)
    print(T.de_tokenize(token_ids_s))
