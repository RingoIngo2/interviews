from collections import Counter


class SimpleTokenizer:
    unknown = "<UNKNOWN>"
    start = "<START>"
    end = "<END>"
    padding = "<PAD>"

    def __init__(self, corpus: str, max_number_tokens: int = 100):
        self.max_number_tokens = max_number_tokens
        self.token_to_id = self._init_token_to_id(corpus)
        self.id_to_token = self._init_id_to_token()

    @property
    def id_start_token(self) -> int:
        return self.token_to_id[self.start]

    @property
    def id_end_token(self) -> int:
        return self.token_to_id[self.end]

    @property
    def id_unknown_token(self) -> int:
        return self.token_to_id[self.unknown]

    @property
    def id_padding_token(self) -> int:
        return self.token_to_id[self.padding]

    def _init_token_to_id(self, corpus: str) -> dict[str, int]:
        counter = Counter(self._split(corpus))
        token_to_id = {self.padding: 0, self.unknown: 1, self.start: 2, self.end: 3}
        token_to_id.update(
            {
                t: i + 4
                for i, (t, _) in enumerate(
                    counter.most_common(self.max_number_tokens - 4)
                )
            }
        )
        return token_to_id

    def _init_id_to_token(self) -> dict[int, str]:
        return {i: t for t, i in self.token_to_id.items()}

    @staticmethod
    def _split(s: str) -> list[str]:
        return s.lower().split()

    def tokenize(self, s: str, length: int = 0) -> list[int]:
        tokens = [self.id_start_token] + self._tokenize(s) + [self.id_end_token]
        if not length:
            return tokens
        if length < len(tokens):
            return tokens[:length]
        return tokens + [self.id_padding_token] * (length - len(tokens))

    def _tokenize(self, s: str) -> list[int]:
        tokens = self._split(s)
        return [self.token_to_id.get(t, self.id_unknown_token) for t in tokens]

    def de_tokenize(self, ids: list[int]) -> str:
        return "".join(self.id_to_token[i] for i in ids)


if __name__ == "__main__":
    s = "Hi, how are you? Whats UPP"
    tokenizer = SimpleTokenizer(corpus=s, max_number_tokens=7)
    print(tokenizer.tokenize(s))
    print(tokenizer.de_tokenize([0, 4, 1, 1]))
    print(tokenizer.tokenize(s, length=15))

