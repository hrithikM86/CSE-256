
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
hidden_size_DAN = 64
lr_DAN = 0.0001
epochs_DAN = 100
embed_path = 'data/glove.6B.50d-relativized.txt'


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

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

        

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)
        x = self.embeddings(x)
        x = (x * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        x = x / lengths

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.log_softmax(x)

        return x