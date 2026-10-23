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
import csv
import os
from BPEtokenizer import BPETokenizer



def save_index_word_map(index_word_map):
    os.makedirs("mappings", exist_ok=True)

    with open("mappings/index_word_map.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "word"])
        writer.writerows(sorted(index_word_map.items()))

def save_dan_results(train_accuracy, dev_accuracy, train_loss, dev_loss, pretrained, filename=None):
    os.makedirs("results", exist_ok=True)
    if filename is None:
        filename = f"results/DAN_PT_{pretrained}.csv"

    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "epoch", "train_accuracy", "dev_accuracy", "train_loss", "dev_loss"
        ])
        writer.writerows(zip(
            range(1, len(train_accuracy) + 1),
            train_accuracy,
            dev_accuracy,
            train_loss,
            dev_loss
        ))

    print(f"\nResults saved as {filename}")

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
@torch.no_grad()
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
@torch.no_grad()
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

def experiment_DAN(
    model,
    train_loader,
    test_loader,
    checkpoint_file="checkpoints/DAN_best.pt"
):
    os.makedirs(os.path.dirname(checkpoint_file) or ".", exist_ok=True)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of paramaters in DAN: {num_params}")

    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_DAN)

    all_train_accuracy = []
    all_test_accuracy = []
    all_train_loss = []
    all_test_loss = []
    best_dev_accuracy = -1.0

    for epoch in range(epochs_DAN):
        train_accuracy, train_loss = train_epoch_DAN(
            train_loader, model, loss_fn, optimizer
        )
        all_train_accuracy.append(train_accuracy)
        all_train_loss.append(train_loss)

        test_accuracy, test_loss = eval_epoch_DAN(
            test_loader, model, loss_fn, optimizer
        )
        all_test_accuracy.append(test_accuracy)
        all_test_loss.append(test_loss)

        if test_accuracy > best_dev_accuracy:
            best_dev_accuracy = test_accuracy

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "dev_accuracy": test_accuracy,
                    "dev_loss": test_loss
                },
                checkpoint_file
            )

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}, train loss {train_loss:.3f}, dev loss {test_loss:.3f}')

    checkpoint = torch.load(
        checkpoint_file,
        map_location=next(model.parameters()).device,
        weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"\nBest model saved as {checkpoint_file}"
        f"\nEpoch: {checkpoint['epoch']}, "
        f"dev accuracy: {checkpoint['dev_accuracy']:.3f}, "
        f"dev loss: {checkpoint['dev_loss']:.3f}"
    )

    return all_train_accuracy, all_test_accuracy, all_train_loss, all_test_loss


def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')

    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')

    parser.add_argument(
    "--pretrained",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use GloVe initialization instead of random embeddings"
    )

    parser.add_argument("--num-merges", type=int, default=15000)

    # Parse the command-line arguments
    args = parser.parse_args()
    if args.num_merges < 0:
        parser.error("--num-merges must be nonnegative")
    os.makedirs("figures", exist_ok=True)

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
        training_accuracy_file = 'figures/train_accuracy.png'
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
        testing_accuracy_file = 'figures/dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()

    elif args.model == "DAN":

            # Data Loading
            start_time = time.time()

            word_embeddings = read_word_embeddings(embeddings_file=embed_path)
            # word_indexer and vectors

            # # Saving index:word mapping
            # save_index_word_map(word_embeddings.word_indexer.ints_to_objs)

            train_data = SentimentDatasetDAN("data/train.txt", word_embeddings=word_embeddings)
            dev_data = SentimentDatasetDAN("data/dev.txt", word_embeddings=word_embeddings)
            # we are returning inices of the words

            train_loader = DataLoader(train_data, batch_size=batch_size_DAN, shuffle=True, collate_fn=collate_dan)
            test_loader = DataLoader(dev_data, batch_size=batch_size_DAN, shuffle=False, collate_fn=collate_dan)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"\nData loaded in : {elapsed_time} seconds")

            start_time = time.time()

            # Train and evaluate DAN
            print('\nDAN layers:')
            dan_train_accuracy, dan_test_accuracy, dan_train_loss, dan_test_loss = experiment_DAN(
                DAN(
                    hidden_size=hidden_size_DAN,
                    word_embeddings=word_embeddings,
                    pretrained=args.pretrained
                ),
                train_loader,
                test_loader,
                checkpoint_file=f"checkpoints/DAN_PT_{args.pretrained}_best.pt"
            )

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"\nModel Trained in : {elapsed_time} seconds")

            # Save results as csv
            save_dan_results(dan_train_accuracy, dan_test_accuracy, dan_train_loss, dan_test_loss, args.pretrained
            )
    
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
            accuracy_file = f'figures/DAN_Accuracy_PT_{args.pretrained}.png'
            plt.savefig(accuracy_file)
            print(f"\nAccuracy plot saved as {accuracy_file}")


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
            loss_file = f'figures/DAN_Loss_PT_{args.pretrained}.png'
            plt.savefig(loss_file)
            print(f"\nLoss plot saved as {loss_file}")
    
            # plt.show()

    elif args.model in ("DAN_BPE", "SUBWORDDAN"):

            # Data Loading
            start_time = time.time()

            tokenizer = BPETokenizer(
                "data/train.txt", num_merges=args.num_merges
            )

            os.makedirs("mappings", exist_ok=True)
            merges_file = f"mappings/BPE_merges_{args.num_merges}.json"
            indexer_file = f"mappings/BPE_indexer_{args.num_merges}.json"

            if os.path.exists(merges_file) and os.path.exists(indexer_file):
                tokenizer.load_merges(merges_file)
                indexer = tokenizer.load_indexer(indexer_file)
            else:
                tokenizer.tokenize()
                indexer = tokenizer.build_subword_indexer()

                tokenizer.save_merges(merges_file)
                tokenizer.save_indexer(indexer_file)
            print(f"Subword vocabulary size: {len(indexer)}")

            train_data = SentimentDatasetSubwordDAN("data/train.txt", tokenizer, indexer)
            dev_data = SentimentDatasetSubwordDAN("data/dev.txt", tokenizer, indexer)

            train_loader = DataLoader(train_data, batch_size=batch_size_DAN, shuffle=True, collate_fn=collate_dan)
            test_loader = DataLoader(dev_data, batch_size=batch_size_DAN, shuffle=False, collate_fn=collate_dan)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"\nData loaded in : {elapsed_time} seconds")

            start_time = time.time()

            # Train and evaluate DAN
            # Train and evaluate DAN
            print('\nDAN BPE layers:')
            dan_train_accuracy, dan_test_accuracy, dan_train_loss, dan_test_loss = experiment_DAN(
                DAN(
                    hidden_size=hidden_size_DAN,
                    pretrained=False,
                    vocab_size=len(indexer),
                    embedding_dim=embedding_dim_bpe
                ),
                train_loader,
                test_loader,
                checkpoint_file=f"checkpoints/DAN_BPE_Merges_{args.num_merges}_best.pt"
            )

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"\nModel Trained in : {elapsed_time} seconds")

            # Save results as csv
            save_dan_results(dan_train_accuracy, dan_test_accuracy, dan_train_loss, dan_test_loss, False,
                             filename=f"results/DAN_BPE_Merges_{args.num_merges}.csv")
    
            # Plot the accuracy
            plt.figure(figsize=(8, 6))
            plt.plot(dan_train_accuracy, label='train')
            plt.plot(dan_test_accuracy, label='test')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title('DAN BPE Epoch vs Accuracy')
            plt.legend()
            plt.grid()
    
            # Save the training accuracy figure
            accuracy_file = f'figures/DAN_BPE_Accuracy_Merges_{args.num_merges}.png'
            plt.savefig(accuracy_file)
            print(f"\nAccuracy plot saved as {accuracy_file}")


            # Plot the loss
            plt.figure(figsize=(8, 6))
            plt.plot(dan_train_loss, label='train')
            plt.plot(dan_test_loss, label='test')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('DAN BPE Epoch vs Loss')
            plt.legend()
            plt.grid()
    
            # Save the training accuracy figure
            loss_file = f'figures/DAN_BPE_Loss_Merges_{args.num_merges}.png'
            plt.savefig(loss_file)
            print(f"\nLoss plot saved as {loss_file}")
    
            # plt.show()


if __name__ == "__main__":
    main()

# python main.py --model BOW
# python main.py --model DAN --pretrained
# python main.py --model DAN --no-pretrained
# python main.py --model DAN_BPE --num-merges 15000
