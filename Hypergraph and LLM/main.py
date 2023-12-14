import pandas as pd
import torch
import networkx as nx
from transformers import BertTokenizer, BertModel
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, GATConv
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import numpy as np
import argparse
import pickle
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings


import pandas as pd
import torch
import networkx as nx
from transformers import BertTokenizer, BertModel
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, GATConv
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import numpy as np
import argparse
import pickle
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore")

# Set a random seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


#Functions to save a python list
def save_list(filename, my_list):
    with open(filename, 'wb') as file:
        pickle.dump(my_list, file)

def load_list(filename):
    with open(filename, 'rb') as file:
        my_list = pickle.load(file)
    return my_list




# Split text into n parts and get BERT embeddings
def get_bert_embeddings(text, n_splits):
    tokenized_text = tokenizer.tokenize(text)
    part_size = int(len(tokenized_text) / n_splits)

    if part_size == 0:
        part_size = 1

    if len(tokenized_text) == 1:
        segments = [tokenized_text, tokenized_text]

    else:
        segments = [tokenized_text[i:i + part_size] for i in range(0, len(tokenized_text), part_size)]
    embeddings = []

    for segment in segments:

        #Some sentence division can have more than the accepted input sequence length for BERT
        if len(segment) >512:
            segment = segment[:512]
        indexed_tokens = tokenizer.convert_tokens_to_ids(segment)
        tokens_tensor = torch.tensor([indexed_tokens]).to(device)
        with torch.no_grad():
            output = bert_model(tokens_tensor)
            pooled_output = output[1]
            pooled_output = pooled_output.detach().cpu()
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
    def __init__(self, num_features, attention_heads):
        super(GATClassifier, self).__init__()
        self.conv1 = GATConv(num_features, 32,  heads=attention_heads)
        self.conv2 = GATConv(32 * attention_heads, 16, heads=attention_heads)
        self.fc = torch.nn.Linear(16 * attention_heads, 32)
        self.fc2 = torch.nn.Linear(32, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = global_add_pool(x, torch.zeros(x.shape[0], dtype=torch.long).to(device))
        x = self.fc(x)
        x = self.fc2(x)

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
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out, data.y)
        total_loss+= loss
        loss.backward()
        optimizer.step()

    total_loss = total_loss / len(train_loader)
    return total_loss


# Evaluation loop
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            loss = loss_fn(out, data.y)
            total_loss += loss
            _, predicted = torch.max(out.detach(), 1)
            total += data.y.size(0)
            correct += (predicted == data.y).sum().item()

    accuracy = correct / total
    total_loss = total_loss / len(loader)
    return accuracy, total_loss

#Function to calcluate all metrics
def evaluate_test(model, loader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    total_preds = []
    ground_truth = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            loss = loss_fn(out, data.y)
            total_loss += loss
            _, predicted = torch.max(out.detach().cpu(), 1)
            total_preds.append(predicted)
            ground_truth.append(data.y.item())
            total += data.y.size(0)
            correct += (predicted == data.y.cpu()).sum().item()

    accuracy = correct / total
    total_preds  = np.concatenate(total_preds, axis=0)
    # ground_truth = np.array(ground_truth)

    precision = precision_score(ground_truth, total_preds)
    recall = recall_score(ground_truth, total_preds)
    f1 = f1_score(ground_truth, total_preds)

    return accuracy, total_loss, precision, recall, f1

#Function to split the dataset
def split_data_training_validation_testing_same_domain(data_list):
    train_data, temp_data = train_test_split(data_list, random_state=0, test_size=0.4)
    val_data, test_data = train_test_split(temp_data, random_state=0, test_size=0.5)

    return train_data, val_data, test_data

#Returns a list called data list which has the data in graph format
def get_data_list(texts, labels, n_splits):
    print("Get Embeddings:")
    embeddings_list = [get_bert_embeddings(text, n_splits) for text in tqdm.tqdm(texts)]

    print("Getting graphs:")
    graphs = [create_graph(embeddings) for embeddings in tqdm.tqdm(embeddings_list)]

    print("Getting Data From Graph:")
    data_list = [graph_to_data(graph, label) for graph, label in tqdm.tqdm(zip(graphs, labels))]

    return data_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainDomain', type = str, default = 'wiki_csai')
    parser.add_argument('--testDomain', type=str, default='medicine')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batchSize', type=int, default=1)
    parser.add_argument('--attentionHeads', type=int, default=4)
    parser.add_argument('--noOfSplits', type=int, default=7)

    #Device to be trained on
    device = "cuda:1"

    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

    #Getting the command line arguments
    args = parser.parse_args()
    epochs = args.epochs
    lr = args.lr
    trainDomain = args.trainDomain
    testDomain = args.testDomain
    batch_size = args.batchSize
    attention_heads = args.attentionHeads
    n_splits = args.noOfSplits

    wandb.init(project="reverse-engineering-cot-cross-domain-experiment-new-parameter-itr1", config=args)

    # Loading the domain csv
    train_df = pd.read_csv(f"data/{trainDomain}_full.csv")
    cross_domain_df = pd.read_csv(f"data/{testDomain}_full.csv")

    # train_df = train_df.loc[:10]  #Code for sample testing
    # cross_domain_df = cross_domain_df.loc[:10]


    # Prepare data for training, testing, and validation in train domain
    train_texts = train_df['answer'].tolist()
    train_labels = train_df['label'].tolist()

    # Prepare data for training, testing, and validation in cross domain
    cross_texts = cross_domain_df['answer'].tolist()
    cross_labels = cross_domain_df['label'].tolist()

    #Converting the text data to proposed graph data
    train_data_list = get_data_list(train_texts, train_labels, n_splits)
    cross_data_list = get_data_list(cross_texts, cross_labels, n_splits)

    # If training and testing domain are same then we have to get the testing data from training data itself.
    if args.trainDomain == args.testDomain:
        train_data, val_data, test_data = split_data_training_validation_testing_same_domain(train_data_list)
    else:
        # The test dataset should be the same across cases.
        train_data, val_data, _ = split_data_training_validation_testing_same_domain(train_data_list)
        _, _, test_data = split_data_training_validation_testing_same_domain(cross_data_list)


    # Define model, optimizer, and loss function
    model = GATClassifier(num_features=768, attention_heads=attention_heads).to(device) # Assuming BERT embeddings have size 768
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = F.cross_entropy

    # Convert data to PyTorch Geometric DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)


    # Train the model

    print("Training the model:")
    wandb.run.name = f"RE_CoT_TrainDomain_{trainDomain}_TestDomain_{testDomain}"
    wandb.watch(model, log="all")
    best_val_acc = 0
    for epoch in tqdm.tqdm(range(epochs)):
        train_loss = train(model, optimizer, loss_fn, train_loader)
        train_acc, _ = evaluate(model, train_loader)
        val_acc, val_loss = evaluate(model, val_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_name = f"RE_COT_trainDomain_{trainDomain}_testDomain_{testDomain}.pth"
            checkpoint = {'epoch':epoch+1,'model_state_dict': model.state_dict(), 'best_val_acc': best_val_acc,
                          'optimizer_state_dict': optimizer.state_dict()}
            torch.save(checkpoint, "models/" + model_name)

        wandb.log({"epoch": epoch + 1, "training_loss": train_loss, "training_accuracy": train_acc, "validation_loss": val_loss,
                   "validation_accuracy": val_acc})

    # Evaluate on test set
    best_model_path = f"models/RE_COT_trainDomain_{trainDomain}_testDomain_{testDomain}.pth"
    checkpoint = torch.load(best_model_path)
    eval_model = GATClassifier(num_features=768, attention_heads=attention_heads).to(device)
    eval_model.load_state_dict(checkpoint['model_state_dict'])
    test_acc, test_loss, precision, recall, f1_score = evaluate_test(eval_model, test_loader)
    wandb.log({"TestingAccuracy": test_acc, "Precision": precision, "Recall": recall, "F1_Score": f1_score})
