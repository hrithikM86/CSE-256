from sentiment_data import read_sentiment_examples
from utils import Indexer
import collections
import json
import re


class BPETokenizer:
    def __init__(self, infile, num_merges):
        self.examples = read_sentiment_examples(infile)
        self.vocab = collections.defaultdict(int)
        self.num_merges = num_merges
        self.merges = []
        self.subword_indexer = None
        self.encoding_cache = {}

    def create_vocab(self):
        self.vocab = collections.defaultdict(int)

        for example in self.examples:
            for word in example.words:
                key = " ".join(list(word) + ["<w>"])
                self.vocab[key] += 1

    def get_stats(self, vocab):
        pairs = collections.defaultdict(int)

        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq

        return pairs

    def merge_vocab(self, pair, v_in):
        v_out = collections.defaultdict(int)
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")

        for word, freq in v_in.items():
            # A callable replacement preserves literal backslashes.
            merged_word = pattern.sub(lambda match: "".join(pair), word)
            v_out[merged_word] += freq

        return dict(v_out)

    def tokenize(self):
        self.create_vocab()
        self.merges = []
        self.encoding_cache.clear()
        self.subword_indexer = None

        for _ in range(self.num_merges):
            pairs = self.get_stats(self.vocab)

            if not pairs:
                break

            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            self.vocab = self.merge_vocab(best, self.vocab)

    def save_merges(self, filename):
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(self.merges, file, ensure_ascii=False, indent=2)

    def load_merges(self, filename):
        with open(filename, "r", encoding="utf-8") as file:
            self.merges = [tuple(pair) for pair in json.load(file)]

        self.encoding_cache.clear()
        self.subword_indexer = None

    def encode(self, words):
        # Return subword token strings, not indices.
        subwords = []

        for word in words:
            if word not in self.encoding_cache:
                encoded_word = " ".join(list(word) + ["<w>"])

                for pair in self.merges:
                    result = self.merge_vocab(pair, {encoded_word: 1})
                    encoded_word = next(iter(result))

                self.encoding_cache[word] = tuple(encoded_word.split())

            subwords.extend(self.encoding_cache[word])

        return subwords

    def build_subword_indexer(self):
        self.subword_indexer = Indexer()
        self.subword_indexer.add_and_get_index("PAD")  # 0
        self.subword_indexer.add_and_get_index("UNK")  # 1

        for example in self.examples:
            for token in self.encode(example.words):
                self.subword_indexer.add_and_get_index(token)

        return self.subword_indexer

    def save_indexer(self, filename):
        tokens = [
            self.subword_indexer.get_object(i)
            for i in range(len(self.subword_indexer))
        ]

        with open(filename, "w", encoding="utf-8") as file:
            json.dump(tokens, file, ensure_ascii=False, indent=2)

    def load_indexer(self, filename):
        with open(filename, "r", encoding="utf-8") as file:
            tokens = json.load(file)

        self.subword_indexer = Indexer()

        for token in tokens:
            self.subword_indexer.add_and_get_index(token)

        return self.subword_indexer


if __name__ == "__main__":
    bpe = BPETokenizer("data/train.txt", num_merges=15000)
    bpe.tokenize()
    bpe.save_merges("data/BPEtokenizer.json")
    bpe.build_subword_indexer()

    print("Learned merges:", len(bpe.merges))
    print("Subword vocabulary size:", len(bpe.subword_indexer))
    print("Example encoding:", bpe.encode(["the", "director"]))