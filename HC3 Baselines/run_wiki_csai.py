import numpy as np
import torch
import tqdm
import argparse
import pandas as pd
import wandb
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoModel, BertTokenizerFast, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


# Set a random seed for reproducibility
seed = 22
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Hyper Parameters
split_ratio = 0.4
batch_size = 6
cross_entropy = nn.CrossEntropyLoss()  # Loss function


# Model Tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# importing bert
bert = AutoModel.from_pretrained('roberta-base')

# Defining the model architecture
class BERT_Arch(nn.Module):

    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert

        # dropout layer

        self.dropout = nn.Dropout(0.1)

        # relu activation function

        self.relu = nn.ReLU()

        # dense layer 1

        self.fc1 = nn.Linear(768, 512)

        # dense layer 2 (Output layer)

        self.fc2 = nn.Linear(512, 2)

        # softmax activation function

        self.softmax = nn.LogSoftmax(dim=1)

    # define the forward pass
    def forward(self, sent_id, mask):
        # pass the inputs to the model

        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)

        x = self.fc1(cls_hs)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)

        # apply softmax activation

        x = self.softmax(x)

        return x


# Function to split the training data as training, validation and testing for same domains
def split_data_training_validation_testing_same_domain(train_df):
    train_text, temp_text, train_labels, temp_labels = train_test_split(train_df['answer'], train_df['label'],
                                                                        random_state=0,
                                                                        test_size=0.4,
                                                                        )

    val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                    random_state=0,
                                                                    test_size=0.5,
                                                                    )
    return train_text, train_labels, val_text, val_labels, test_text, test_labels


# Function to create a dataset loader
def create_dataset_loader(train_text, train_labels, val_text, val_labels, test_text, test_labels, train_sequence_length,
                          test_sequence_length, batch_size=6):
    tokens_train = tokenizer.batch_encode_plus(
        train_text.tolist(),
        max_length=train_sequence_length,
        pad_to_max_length=True,
        truncation=True
    )

    tokens_val = tokenizer.batch_encode_plus(
        val_text.tolist(),
        max_length=train_sequence_length,
        pad_to_max_length=True,
        truncation=True
    )

    tokens_test = tokenizer.batch_encode_plus(
        test_text.tolist(),
        max_length=test_sequence_length,
        pad_to_max_length=True,
        truncation=True
    )

    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(train_labels.tolist())

    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])
    val_y = torch.tensor(val_labels.tolist())

    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])
    test_y = torch.tensor(test_labels.tolist())

    train_data = TensorDataset(train_seq, train_mask, train_y)

    # sampler for sampling the data during training
    train_sampler = RandomSampler(train_data)

    # dataLoader for train set
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # wrap tensors
    val_data = TensorDataset(val_seq, val_mask, val_y)

    # sampler for sampling the data during training
    val_sampler = SequentialSampler(val_data)

    # dataLoader for validation set
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    # wrap tensors
    test_data = TensorDataset(test_seq, test_mask, test_y)

    # sampler for sampling the data during testing
    test_sampler = SequentialSampler(test_data)

    # dataLoader for test set
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=1)

    return train_dataloader, val_dataloader, test_dataloader


# function to train the model
def train(train_dataloader):
    model.train()

    total_loss, total_accuracy = 0, 0

    # empty list to save model predictions

    total_preds = []

    # iterate over batches

    for step, batch in enumerate(train_dataloader):
        # push the batch to gpu

        batch = [r.to(device) for r in batch]

        sent_id, mask, labels = batch

        # clear previously calculated gradients

        model.zero_grad()

        # get model predictions for the current batch

        preds = model(sent_id, mask)

        # compute the loss between actual and predicted values

        loss = cross_entropy(preds, labels)

        # add on to the total loss

        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients

        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters

        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU

        preds = preds.detach().cpu().numpy()

        # append the model predictions

        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)
    # returns the loss and predictions

    return avg_loss, total_preds


# Function to evaluate the model
def evaluate(val_dataloader):
    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions

    total_preds = []

    # iterate over batches

    for step, batch in enumerate(val_dataloader):
        # push the batch to gpu
        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():
            # model predictions
            preds = model(sent_id, mask)

            # compute the validation loss between actual and predicted values

            loss = cross_entropy(preds, labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

    # compute the validation loss of the epoch

    avg_loss = total_loss / len(val_dataloader)

    # reshape the predictions in form of (number of samples, no. of classes)

    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds


# Function to test the model
def test(test_dataloader, model):
    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions

    total_preds = []
    total_ground_truth = []
    # iterate over batches

    for step, batch in enumerate(test_dataloader):
        # push the batch to gpu
        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():
            # model predictions
            preds = model(sent_id, mask)

            # compute the validation loss between actual and predicted values

            loss = cross_entropy(preds, labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)
            total_ground_truth.append(labels)

    # compute the testing loss of each batch

    avg_loss = total_loss / len(test_dataloader)

    # reshape the predictions in form of (number of samples, no. of classes)

    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainDomain', type=str, default="wiki_csai")
    parser.add_argument('--testDomain', type=str, default="wiki_csai")
    parser.add_argument('--trainSeqLength', type=int, default="512")
    parser.add_argument('--testSeqLength', type=int, default="512")
    parser.add_argument('--epochs', type=int, default="3")
    parser.add_argument('--lr', type=float, default="1e-5")

    args = parser.parse_args()
    trainSequenceLength = args.trainSeqLength
    testSequenceLength = args.testSeqLength
    epochs = args.epochs
    lr = args.lr

    # log the experiment settings
    wandb.init(project="hc3-baseline-roberta-hyperparameter-search-wiki-csai", config=args)

    print("Experiment Details")
    print("Training Data: " + str(args.trainDomain) + " Testing Data: " + str(
        args.testDomain) + " Training Sequence Length: " + str(args.trainSeqLength) + " Testing Sequence Length: "
          + str(args.testSeqLength))

    # Loading the domain csv
    train_df = pd.read_csv(f"data/{args.trainDomain}_full.csv")
    cross_domain_df = pd.read_csv(f"data/{args.testDomain}_full.csv")

    # If training and testing domain are same then we have to get the testing data from training data itself.
    if args.trainDomain == args.testDomain:
        train_text, train_labels, val_text, val_labels, test_text, test_labels = split_data_training_validation_testing_same_domain(
            train_df)
    else:
        # The test dataset should be the same across cases.
        train_text, train_labels, val_text, val_labels, _, _ = split_data_training_validation_testing_same_domain(
            train_df)
        _, _, _, _, test_text, test_labels = split_data_training_validation_testing_same_domain(cross_domain_df)

    train_dataloader, val_dataloader, test_dataloader = create_dataset_loader(train_text, train_labels, val_text,
                                                                              val_labels, test_text, test_labels,
                                                                              trainSequenceLength, testSequenceLength,
                                                                              batch_size)

    # pass the pre-trained BERT to our define architecture
    model = BERT_Arch(bert)
    device = torch.device("cuda")

    # push the model to GPU
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)  # optimizer and learning rate

    # Training and Validation Pipeline
    best_valid_acc = -1

    # empty lists to store training and validation loss of each epoch
    train_losses = []
    valid_losses = []

    # Define hyperparameters to Wandb
    config = wandb.config
    config.learning_rate = lr
    config.num_epochs = epochs
    config.batch_size = batch_size

    # Wandb for different models
    wandb.run.name = "robert_train_" + str(args.trainDomain) + "_test_" + str(args.testDomain) + "_trainSeqLen_" + str(
        trainSequenceLength) + "_testSeqLen_" + str(testSequenceLength)

    wandb.watch(model, log="all")

    # for each epoch

    for epoch in tqdm.tqdm(range(epochs)):
        # train model
        train_loss, _ = train(train_dataloader)
        valid_loss, val_preds = evaluate(val_dataloader)

        # Calculating valdiation accuracy
        preds_val = np.argmax(val_preds, axis=1)
        val_y = torch.tensor(val_labels.tolist())
        val_report = classification_report(val_y, preds_val, output_dict=True)
        valid_accuracy = val_report['accuracy']

        # save the best model
        if valid_accuracy > best_valid_acc:
            best_valid_acc = valid_accuracy
            model_name = "robert_train_" + str(args.trainDomain) + "_test_" + str(
                args.testDomain) + "_trainSeqLen_" + str(trainSequenceLength) + "_testSeqLen_" + str(
                testSequenceLength) + ".pt"
            checkpoint = {'state_dict': model.state_dict(), 'best_val_acc': best_valid_acc,
                          'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, "models/" + model_name)
        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        wandb.log({"epoch": epoch + 1, "training_loss": train_loss, "validation_loss": valid_loss,
                   "validation_accuracy": valid_accuracy})

    best_model_path = "models/robert_train_" + str(args.trainDomain) + "_test_" + str(
                args.testDomain) + "_trainSeqLen_" + str(trainSequenceLength) + "_testSeqLen_" + str(
                testSequenceLength) + ".pt"
    checkpoint = torch.load(best_model_path)
    eval_model = BERT_Arch(bert).to(device)
    eval_model.load_state_dict(checkpoint['state_dict'])
    average_loss, total_preds = test(test_dataloader, eval_model)  # Testing

    # Getting the metrics
    preds = np.argmax(total_preds, axis=1)
    test_y = torch.tensor(test_labels.tolist())
    report = classification_report(test_y, preds, output_dict=True)

    macro_precision = report['macro avg']['precision']
    macro_recall = report['macro avg']['recall']
    macro_f1 = report['macro avg']['f1-score']

    accuracy = report['accuracy']

    # Logging to Wandb
    wandb.log({"test_accuracy": accuracy, "precision": macro_precision, "recall": macro_recall, "f1-score": macro_f1})








