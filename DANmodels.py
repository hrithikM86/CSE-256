import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples, read_word_embeddings
from torch.utils.data import Dataset
import numpy as np

# DAN Paramaters
batch_size_DAN = 16
hidden_size_DAN = 128
lr_DAN = 0.0001
epochs_DAN = 100
embed_path = '/Users/hrithik/Desktop/UC San Diego/Courses/UCSD CSE 256/CSE256_PA1_FA26/data/glove.6B.50d-relativized.txt'

class DAN(nn.Module):
    def __init__(self, hidden_size, word_embeddings):
        super().__init__()
        self.embeddings = word_embeddings.get_initialized_embedding_layer(frozen = True)
        input_size = word_embeddings.get_embedding_length()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,2)
        self.log_softmax = nn.LogSoftmax(dim = 1)

        

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


