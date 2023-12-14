import pandas as pd
import argparse
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='finance')
    args = parser.parse_args()
    domain = args.domain

    data = pd.read_csv(f"../data/{domain}_full.csv")


    # Load a pre-trained BERT model for sentence embedding
    model = SentenceTransformer('bert-base-nli-mean-tokens')

    # Get the sentence embeddings for the generated_text column
    embeddings = model.encode(data['answer'].tolist())

    # Perform t-SNE to reduce the dimensionality of the embeddings
    tsne = TSNE(n_components=2, perplexity=50, random_state=10)
    tsne_embeddings = tsne.fit_transform(embeddings)

    # Assign colors to different labels
    label_to_color = {
        0: 'red',
        1: 'blue',
    }

    # Create a list of colors for each data point based on its label
    colors = [label_to_color[label] for label in data['label']]

    # Create a scatter plot with the t-SNE transformed data
    plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=colors)

    # Add a legend with the label names and corresponding colors
    handles = [plt.plot([], [], marker="o", ls="", color=color, label=label)[0] for label, color in
               label_to_color.items()]
    plt.legend(handles=handles)

    plt.title(f"{domain} full data t-SNE")
    plt.savefig(f"{domain}_tsne.png")
    plt.show()
    print(f"Domain: {domain}, File Saved Sucessfully")