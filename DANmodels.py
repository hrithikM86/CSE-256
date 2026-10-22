import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples, read_word_embeddings
from torch.utils.data import Dataset
import numpy as np

# DAN Paramaters
embedding_dim_bpe = 50
batch_size_DAN = 16
hidden_size_DAN = 100
lr_DAN = 0.0001
epochs_DAN = 100
embed_path = 'data/glove.6B.300d-relativized.txt'

class DAN(nn.Module):
    def __init__(
        self,
        hidden_size,
        word_embeddings=None,
        pretrained=True,
        vocab_size=None,
        embedding_dim=50
    ):
        super().__init__()

        if word_embeddings is not None:
            input_size = word_embeddings.get_embedding_length()
            vocab_size = len(word_embeddings.word_indexer)
        else:
            input_size = embedding_dim

        if pretrained:
            if word_embeddings is None:
                raise ValueError("Pretrained mode requires word_embeddings")

            self.embeddings = (
                word_embeddings.get_initialized_embedding_layer(frozen=True)
            )
        else:
            if vocab_size is None:
                raise ValueError("Random embeddings require vocab_size")

            self.embeddings = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=input_size,
                padding_idx=0
            )


        # def hidden_block(in_features, out_features):
        #     return nn.Sequential(
        #         nn.Linear(in_features, out_features),
        #         nn.LayerNorm(out_features),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.3)
        #     )

        # self.layer1 = hidden_block(input_size, hidden_size)
        # self.layer2 = hidden_block(hidden_size, hidden_size)
        # self.layer3 = hidden_block(hidden_size, hidden_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(p = 0.5)
        self.log_softmax = nn.LogSoftmax(dim=1)
        

        

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)
        x = self.embeddings(x)
        x = (x * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        x = x / lengths

        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc2(x)) 
        # x = self.dropout(x)
        x = self.fc3(x)
        x = self.log_softmax(x)

        return x



class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, word_embeddings):

        self.examples = read_sentiment_examples(infile) #Read and tokanize sentences. Each sample is a class with words, label
        indexer = word_embeddings.word_indexer

        
        self.word_indices = [
            torch.tensor(
                [
                    indexer.index_of(word)
                    if indexer.contains(word)
                    else indexer.index_of("UNK")
                    for word in example.words
                ],
                dtype=torch.long
            )
            for example in self.examples
        ]

        self.labels = torch.tensor([example.label for example in self.examples], dtype=torch.long)

    def __len__(self):
            return len(self.examples)

    def __getitem__(self, idx):
            return self.word_indices[idx], self.labels[idx]


class SentimentDatasetSubwordDAN(Dataset):
    def __init__(self, infile, tokenizer, indexer):
        self.examples = read_sentiment_examples(infile)
        unk_index = indexer.index_of("UNK")

        self.word_indices = []

        for example in self.examples:
            tokens = tokenizer.encode(example.words)

            indices = [
                indexer.index_of(token)
                if indexer.contains(token)
                else unk_index
                for token in tokens
            ]

            self.word_indices.append(
                torch.tensor(indices, dtype=torch.long)
            )

        self.labels = torch.tensor(
            [example.label for example in self.examples],
            dtype=torch.long
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.word_indices[idx], self.labels[idx]