import pandas as pd
import torch
import networkx as nx
from transformers import BertTokenizer, BertModel
from torch_geometric.nn import global_add_pool
from sklearn.model_selection import train_test_split
from torch_geometric.nn import HypergraphConv
from transformers import AdamW
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import numpy as np
import argparse
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
import itertools

device = "cuda:0"
warnings.filterwarnings("ignore")

#Defining BERT Model
bert_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name).to(device)


# Split text into n parts and get BERT embeddings
def get_bert_embeddings(text, n_splits):
    tokenized_text = tokenizer.tokenize(str(text))
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

#Get Full Connected Hypergraph Edges
def generate_hyperedges(nodes):
    hyperedges = []
    n = nodes
    nodes = list(range(0, n))

    for r in range(2, n+1):  # Generate hyperedges of size 2 to n
        hyperedges.extend(itertools.combinations(nodes, r))

    return hyperedges
def convert_to_tensor(data):
    flattened_data = [item for sublist in data for item in sublist]
    number_list = []
    for idx, i in enumerate(data):
        l = len(i)
        for j in range(l):
            number_list.append(idx)
    tensor_data = torch.tensor([flattened_data, number_list, ]).to(device)
    return tensor_data


# Convert graph to PyTorch Geometric Data object
def graph_to_data_tensors(graph, label):
    no_of_nodes = len(graph.nodes)
    hyperedge_list = generate_hyperedges(no_of_nodes)
    edge_index = convert_to_tensor(hyperedge_list)
    x = torch.tensor([np.array(graph.nodes[i]['embedding']) for i in graph.nodes], dtype=torch.float).to(device)
    y = torch.tensor(label, dtype=torch.long).to(device)
    return x, edge_index, y

# Function to split the training data as training, validation and testing for same domains for BERT
def split_data_training_validation_testing_same_domain_bert(train_df):
    train_text, temp_text, train_labels, temp_labels = train_test_split(train_df['answer'], train_df['label'],
                                                                        random_state=0,
                                                                        test_size=0.4,
                                                                        )

    val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                    random_state=0,
                                                                    test_size=0.5,
                                                                    )
    return train_text, train_labels, val_text, val_labels, test_text, test_labels

#Model Definition
class HyperGraph(torch.nn.Module):
    def __init__(self, num_features, attention_heads):
        super(HyperGraph, self).__init__()
        self.conv1 = HypergraphConv(num_features, 32, use_attention = "True", attention_mode = "node", heads=attention_heads)
        self.conv2 = HypergraphConv(32 * attention_heads, 16)

    def forward(self, x, edge_index,  hyper_atrr):
        x = self.conv1(x, edge_index,  hyperedge_attr = hyper_atrr)
        x = torch.relu(x)
        x = self.conv2(x, edge_index,  hyperedge_attr = hyper_atrr)
        x = global_add_pool(x, torch.zeros(x.shape[0], dtype=torch.long).to(device))
        return x

#Model Definition
class HyperGraphAndLLM(torch.nn.Module):
    def __init__(self, num_features, attention_heads):
        super(HyperGraphAndLLM, self).__init__()
        self.hypergraph = HyperGraph(num_features, attention_heads)
        self.bert_model = bert_model
        hidden_dim = 16
        self.classifier = torch.nn.Linear(hidden_dim + bert_model.config.hidden_size, 2)


    def forward(self, text, label):
        embedding = get_bert_embeddings(text, n_splits)
        graph = create_graph(embedding)
        x, edge_index, y = graph_to_data_tensors(graph, label)

        n = edge_index.size()[1]
        hyper_atrr = torch.ones(n, 768).to(device)


        graph_output = self.hypergraph(x,edge_index, hyper_atrr)

        bert_input_ids = tokenizer(text, return_tensors='pt', padding=True, truncation=True)['input_ids'].to(device)
        bert_output = self.bert_model(input_ids=bert_input_ids)[1]  # Get the [CLS] token output

        # Concatenate graph and BERT outputs
        combined_output = torch.cat((graph_output, bert_output), dim=1)

        # Pass through classifier layer for binary classification
        output = self.classifier(combined_output)

        return output, y

from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]
        return text, label

# Function to create a dataset loader
def create_dataset_loader(train_text, train_labels, val_text, val_labels, test_text, test_labels, batch_size=1):

    train_dataset = CustomDataset(train_text, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    val_dataset = CustomDataset(val_text, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    test_dataset = CustomDataset(test_text, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader

def train(model, bert_optimizer, graph_optimizer, loss_fn, train_dataloader, flag):
    model.train()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        batch_texts, batch_labels = batch
        text = batch_texts[0]
        label = batch_labels[0]
        bert_optimizer.zero_grad()
        graph_optimizer.zero_grad()
        out, y = model(text, label)
        loss = loss_fn(out.squeeze(0), y)
        total_loss += loss
        loss.backward()

        if flag == True:
            bert_optimizer.step()
            print("BERT Weights Are Updated")
            flag = False
        graph_optimizer.step()

    total_loss = total_loss / len(train_dataloader)
    return total_loss

def evaluate(model, bert_optimizer, graph_optimizer, loss_fn, train_dataloader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    total_preds = []
    ground_truth = []

    with torch.no_grad():
        for batch_texts, batch_labels in train_dataloader:
            bert_optimizer.zero_grad()
            graph_optimizer.zero_grad()
            out, y = model(batch_texts, batch_labels)
            loss = loss_fn(out, y)
            total_loss += loss
            _, predicted = torch.max(out.detach().cpu(), 1)
            total_preds.append(predicted)
            ground_truth.append(y.item())
            total += y.size(0)
            correct += (predicted == y.cpu()).sum().item()

    accuracy = correct / total
    total_preds = np.concatenate(total_preds, axis=0)
    # ground_truth = np.array(ground_truth)

    precision = precision_score(ground_truth, total_preds)
    recall = recall_score(ground_truth, total_preds)
    f1 = f1_score(ground_truth, total_preds)
    total_loss = total_loss / len(train_dataloader)

    return accuracy, total_loss, precision, recall, f1


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
    device = "cuda:0"

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

    wandb.init(project="hypergrahllm-iteration-1", config=args)

    print("Experiment Detail:")
    print(f"TrainDomain:{trainDomain}  Test Domain:{testDomain}")

    # Loading the domain csv
    train_df = pd.read_csv(f"data/{trainDomain}_full.csv")
    cross_domain_df = pd.read_csv(f"data/{testDomain}_full.csv")

    # train_df = train_df.loc[:50]  #Code for sample testing
    # cross_domain_df = cross_domain_df.loc[:50]

    # If training and testing domain are same then we have to get the testing data from training data itself.
    if args.trainDomain == args.testDomain:
        train_text, train_labels, val_text, val_labels, test_text, test_labels = split_data_training_validation_testing_same_domain_bert(train_df)

    else:
        # The test dataset should be the same across cases.
        train_text, train_labels, val_text, val_labels, _, _ = split_data_training_validation_testing_same_domain_bert(
            train_df)
        _, _, _, _, test_text, test_labels = split_data_training_validation_testing_same_domain_bert(cross_domain_df)

    train_dataloader, val_dataloader, test_dataloader = create_dataset_loader(train_text, train_labels, val_text,
                                                                              val_labels, test_text, test_labels,
                                                                              batch_size)

    # print(type(train_text))
    # for step, batch in enumerate(train_dataloader):
    #     print(batch)

    # print(train_dataloader)

    model = HyperGraphAndLLM(num_features=768, attention_heads=attention_heads).to(device)
    bert_optimizer = AdamW(model.bert_model.parameters(), lr=1e-5)
    graph_optimizer = optim.Adam(model.hypergraph.parameters(), lr=lr)
    loss_fn = F.cross_entropy

    print("Training the model:")
    wandb.run.name = f"HypergraphLLM_TrainDomain_{trainDomain}_TestDomain_{testDomain}"
    wandb.watch(model, log="all")
    best_val_acc= 0
    flag = False
# Training the model
    for epoch in tqdm.tqdm(range(epochs)):
        if epoch%30 == 0:
            flag = True
        train_loss = train(model,bert_optimizer,graph_optimizer,loss_fn,train_dataloader, flag)
        flag = False
        train_acc, _, _, _, _ = evaluate(model,bert_optimizer,graph_optimizer,loss_fn,train_dataloader)
        val_acc, val_loss, _, _, _ = evaluate(model, bert_optimizer,graph_optimizer,loss_fn, val_dataloader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_name = f"HypergraphLLM_trainDomain_{trainDomain}_testDomain_{testDomain}.pth"
            checkpoint = {'epoch':epoch+1,'model_state_dict': model.state_dict(), 'best_val_acc': best_val_acc,
                          'optimizer_state_dict': bert_optimizer.state_dict()}
            torch.save(checkpoint, "models/" + model_name)

        wandb.log({"epoch": epoch + 1, "training_loss": train_loss, "training_accuracy": train_acc, "validation_loss": val_loss,
                   "validation_accuracy": val_acc})

    # Evaluate on test set
    best_model_path = f"models/HypergraphLLM_trainDomain_{trainDomain}_testDomain_{testDomain}.pth"
    checkpoint = torch.load(best_model_path)
    eval_model = HyperGraphAndLLM(num_features=768, attention_heads=attention_heads).to(device)
    eval_model.load_state_dict(checkpoint['model_state_dict'])
    test_acc, test_loss, precision, recall, f1_score = evaluate(eval_model, bert_optimizer,graph_optimizer,loss_fn, test_dataloader)
    wandb.log({"TestingAccuracy": test_acc, "Precision": precision, "Recall": recall, "F1_Score": f1_score})


