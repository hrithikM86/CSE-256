from sentiment_data import read_sentiment_examples
import re, collections, json



class BPETokenizer:
    def __init__(self, infile, num_merges):
        self.examples = read_sentiment_examples(infile)
        self.vocab = collections.defaultdict(int)
        self.num_merges = num_merges


    def create_vocab(self):
        for example in self.examples:
            for word in example.words:
                key = ' '.join(list(word) + ['<w>'])
                self.vocab[key] = self.vocab.get(key, 0) + 1


    def get_stats(self, vocab):
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs

    def merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    def tokenize(self):
        self.vocab = collections.defaultdict(int)
        self.create_vocab()
        self.merges = []

        for i in range(self.num_merges):
            pairs = self.get_stats(self.vocab)

            if not pairs:
                break

            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            self.vocab = self.merge_vocab(best, self.vocab)
            # print(best)

        # return self.merges

    def save_merges(self, filename):
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(self.merges, file, ensure_ascii=False, indent=2)

    def load_merges(self, filename):
        with open(filename, "r", encoding="utf-8") as file:
            self.merges = [tuple(pair) for pair in json.load(file)]


    def encode(self, words):
        # Apply learned merges and return subword indices.
        pass




BPE = BPETokenizer("data/train.txt", 15000)
# BPE.tokenize()
# BPE.save_merges("data/BPEtokenizer.json")
BPE.load_merges("data/BPEtokenizer.json")
print(BPE.merges)