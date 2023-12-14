import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif

domain = 'finance'
df = pd.read_csv(f"../data/{domain}_full.csv")

# Calculate mutual information for each label separately
labels = df['label'].unique()
top_10_words = []

for label in labels:
    label_df = df[df['label'] == label]

    # Convert 'answer' column to a bag-of-words representation
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(label_df['answer'])

    # Calculate mutual information between 'answer' and current label
    mi = mutual_info_classif(X, label_df['label'])
    print(mi)

    # Create a DataFrame to store words and their mutual information scores
    mi_df = pd.DataFrame({'word': vectorizer.get_feature_names_out(), 'mi_score': mi})

    # Sort DataFrame by mutual information score in descending order
    mi_df = mi_df.sort_values('mi_score', ascending=False)

    # Get the top 10 words affecting the current label
    top_words = mi_df.head(10)['word'].tolist()
    top_10_words.append((label, top_words))

# Print the top 10 words affecting each label
print(f"Domain: {domain}")
for label, words in top_10_words:
    print(f"Top 10 words affecting '{label}':")
    print(words)
