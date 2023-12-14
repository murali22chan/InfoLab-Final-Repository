import pandas as pd
import torch
import networkx as nx
from transformers import BertTokenizer, BertModel
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, GraphConv, GATConv
from torch_geometric.utils import to_networkx
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import numpy as np
import sys
import argparse
import pickle
import wandb

#Functions to save a python list
def save_list(filename, my_list):
    with open(filename, 'wb') as file:
        pickle.dump(my_list, file)

def load_list(filename):
    with open(filename, 'rb') as file:
        my_list = pickle.load(file)
    return my_list


# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# # Sample DataFrame
# df = pd.read_csv("data/finance_full.csv")
# df['answer'] = df['answer'].str.replace('\t', ' ').str.replace('\n', '')
# # df = df.loc[:99] #Code for sample testing

# Split text into 5 parts and get BERT embeddings
def get_bert_embeddings(text):
    tokenized_text = tokenizer.tokenize(text)
    part_size = int(len(tokenized_text) / 5)

    if part_size == 0:
        part_size = 1

    if len(tokenized_text) == 1:
        segments = [tokenized_text, tokenized_text]

    else:
        segments = [tokenized_text[i:i + part_size] for i in range(0, len(tokenized_text), part_size)]
    embeddings = []

    for segment in segments:
        indexed_tokens = tokenizer.convert_tokens_to_ids(segment)
        tokens_tensor = torch.tensor([indexed_tokens])
        with torch.no_grad():
            output = bert_model(tokens_tensor)
            pooled_output = output[1]
            embeddings.append(pooled_output)

    return embeddings


# Create graph with BERT embeddings
def create_graph(embeddings):
    graph = nx.complete_graph(len(embeddings))

    for i, emb in enumerate(embeddings):
        graph.nodes[i]['embedding'] = emb.numpy().squeeze()

    return graph


# Apply Graph Attention Network (GAT) for graph classification
class GATClassifier(torch.nn.Module):
    def __init__(self, num_features):
        super(GATClassifier, self).__init__()
        self.conv1 = GATConv(num_features, 32, heads=4)
        self.conv2 = GATConv(32 * 4, 16, heads=4)
        self.fc = torch.nn.Linear(16 * 4, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = global_add_pool(x, torch.zeros(x.shape[0], dtype=torch.long))
        x = self.fc(x)
        return x


# Convert graph to PyTorch Geometric Data object
def graph_to_data(graph, label):
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    x = torch.tensor([np.array(graph.nodes[i]['embedding']) for i in graph.nodes], dtype=torch.float)
    y = torch.tensor(label, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

# Training loop
def train(model, optimizer, loss_fn, train_loader):
    model.train()

    total_loss = 0

    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out, data.y)
        total_loss+= loss
        loss.backward()
        optimizer.step()
    return total_loss


# Evaluation loop
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index)
            loss = loss_fn(out, data.y)
            total_loss += loss
            _, predicted = torch.max(out.detach(), 1)
            total += data.y.size(0)
            correct += (predicted == data.y).sum().item()

    accuracy = correct / total
    return accuracy, total_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.01)

    args = parser.parse_args()
    epochs = args.epochs
    lr = args.lr

    wandb.init(project="reverse-engineering-cot-hyperparameter-search", config=args)



    print("Loading Data List:")
    data_list = load_list("data_list_v2.pkl")

    train_data, test_data = train_test_split(data_list, test_size=0.2)
    train_data, val_data = train_test_split(train_data, test_size=0.2)

    # Define model, optimizer, and loss function
    model = GATClassifier(num_features=768)  # Assuming BERT embeddings have size 768
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = F.cross_entropy

    # Convert data to PyTorch Geometric DataLoader
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)


    # Train the model

    print("Training the model:")
    wandb.run.name = f"RE_CoT_Epochs_{epochs}_LR_{lr}"
    wandb.watch(model, log="all")
    for epoch in tqdm.tqdm(range(epochs)):
        train_loss = train(model, optimizer, loss_fn, train_loader)
        train_acc, _ = evaluate(model, train_loader)
        val_acc, val_loss = evaluate(model, val_loader)
        wandb.log({"epoch": epoch + 1, "training_loss": train_loss, "training_accuracy": train_acc, "validation_loss": val_loss,
                   "validation_accuracy": val_acc})

    # Evaluate on test set
    test_acc, test_loss = evaluate(model, test_loader)
    wandb.log({"TestingAccuracy": test_acc})
