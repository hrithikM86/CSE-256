# models.py

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples, read_word_embeddings
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from DANmodels import *
from torch.nn.utils.rnn import pad_sequence

def collate_dan(batch):
    sentences, labels = zip(*batch)
    sentences = pad_sequence(
        sentences, batch_first=True, padding_value=0
    )
    return sentences, torch.stack(labels)

# Training function BOW
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function BOW
def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss



# Training function DAN
def train_epoch_DAN(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        # X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss

# Evaluation function DAN
def eval_epoch_DAN(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        # X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}, train loss {train_loss:.3f}, dev loss {test_loss:.3f}')
    
    return all_train_accuracy, all_test_accuracy

def experiment_DAN(model, train_loader, test_loader):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_DAN)

    all_train_accuracy = []
    all_test_accuracy = []
    all_train_loss = []
    all_test_loss = []
    for epoch in range(epochs_DAN):
        train_accuracy, train_loss = train_epoch_DAN(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)
        all_train_loss.append(train_loss)

        test_accuracy, test_loss = eval_epoch_DAN(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)
        all_test_loss.append(test_loss)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}, train loss {train_loss:.3f}, dev loss {test_loss:.3f}')
    
    return all_train_accuracy, all_test_accuracy, all_train_loss, all_test_loss


def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Check if the model type is "BOW"
    if args.model == "BOW":

        # Load dataset
        start_time = time.time()

        train_data = SentimentDatasetBOW("data/train.txt")
        dev_data = SentimentDatasetBOW("data/dev.txt", vectorizer=train_data.vectorizer, train=False)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")

    
        # Train and evaluate NN2
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='2 layers')
        plt.plot(nn3_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_test_accuracy, label='2 layers')
        plt.plot(nn3_test_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()

    elif args.model == "DAN":

            # Data Loading
            start_time = time.time()

            word_embeddings = read_word_embeddings(embeddings_file=embed_path)
            # word_indexer and vectors

            train_data = SentimentDatasetDAN("data/train.txt", word_embeddings=word_embeddings)
            dev_data = SentimentDatasetDAN("data/dev.txt", word_embeddings=word_embeddings)
            # we are returning inices of the words

            train_loader = DataLoader(train_data, batch_size=batch_size_DAN, shuffle=True, collate_fn=collate_dan)
            test_loader = DataLoader(dev_data, batch_size=batch_size_DAN, shuffle=False, collate_fn=collate_dan)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Data loaded in : {elapsed_time} seconds")

            # Train and evaluate DAN
            print('\nDAN layers:')
            dan_train_accuracy, dan_test_accuracy, dan_train_loss, dan_test_loss = experiment_DAN(DAN(hidden_size=hidden_size_DAN, word_embeddings=word_embeddings), train_loader, test_loader)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Data loaded in : {elapsed_time} seconds")
    
            # Plot the accuracy
            plt.figure(figsize=(8, 6))
            plt.plot(dan_train_accuracy, label='train')
            plt.plot(dan_test_accuracy, label='test')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title('DAN Epoch vs Accuracy')
            plt.legend()
            plt.grid()
    
            # Save the training accuracy figure
            accuracy_file = 'DAN_accuracy.png'
            plt.savefig(accuracy_file)
            print(f"\n\n Accuracy plot saved as {accuracy_file}")


            # Plot the loss
            plt.figure(figsize=(8, 6))
            plt.plot(dan_train_loss, label='train')
            plt.plot(dan_test_loss, label='test')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('DAN Epoch vs Loss')
            plt.legend()
            plt.grid()
    
            # Save the training accuracy figure
            loss_file = 'DAN_loss.png'
            plt.savefig(loss_file)
            print(f"\n\n Loss plot saved as {loss_file}")
    
            # plt.show()


if __name__ == "__main__":
    main()
