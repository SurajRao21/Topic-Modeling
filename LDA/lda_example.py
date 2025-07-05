# Required installations (if not already installed):
# !pip install nltk scikit-learn pandas numpy matplotlib seaborn

import re
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# Step 0: Download necessary NLTK resources
# These are required for text preprocessing: lemmatization and stopword removal.
# You only need to run these once (unless you clear your NLTK cache).
nltk.download('wordnet')
nltk.download('stopwords')

# Step 1: Input Documents
# Here’s a small set of messy, unstructured text documents that simulate real-world noise,
# including punctuation, mixed casing, symbols, etc. We'll clean these up before modeling.
documents = [
    "TESLA!!! has announced its *NEW* electric car model...which is expected to 100% revolutionize the auto industry.",
    "The stock-market experienced a significant DROP, due to recent ECONOMIC uncertainty?!",
    "Scientists have discovered a *NEW* species of dino...saur??? in the desert-region of North America.",
    "Tech-industry continues to grow...RAPIDLY, with new startups emerging... especially in #AI & machine-learning.",
    "NASA...successfully launched 1 new satellite: aimed at exploring distant planets--awesome!"
]

# Step 2: Preprocessing Function 
# This function does several things:
# 1. Lowercases the text for uniformity.
# 2. Removes punctuation and tokenizes into words.
# 3. Removes common English stopwords (like 'the', 'is', 'and').
# 4. Lemmatizes each word (reduces words to their base form).
# 5. Returns a cleaned, space-joined string of meaningful tokens.
def clean_text(doc):
    doc = doc.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(doc)
    lemmatizer = WordNetLemmatizer()
    cleaned = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stopwords.words('english') and len(token) > 2
    ]
    return ' '.join(cleaned)

# Apply cleaning to all documents
cleaned_documents = [clean_text(doc) for doc in documents]

# Step 3: Text Vectorization 
# We're converting the cleaned text into a document-term matrix (DTM).
# Each row is a document, and each column is a word (token) from our vocabulary.
# The values in the matrix represent the count of each word in each document.
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(cleaned_documents)
feature_names = vectorizer.get_feature_names_out()

# Step 4: Fit LDA Topic Model
# We now apply Latent Dirichlet Allocation (LDA) to the DTM.
# n_components sets the number of topics we want the model to find.
# LDA learns how likely each word is to belong to each topic.
n_topics = 2
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(X)

# Step 5: Print Top Words for Each Topic
# For each topic, we print out the top 10 words with the highest weights.
# These words give us an idea of what each topic is "about."
n_top_words = 10
for topic_idx, topic in enumerate(lda.components_):
    print(f"\nTopic #{topic_idx + 1}:")
    top_features = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    print(", ".join(top_features))

# Step 6: Document-Topic Distribution
# After fitting the model, we can see how each document relates to the topics.
# This shows us the probability of each topic in each document.
doc_topic_dists = lda.transform(X)
doc_topic_df = pd.DataFrame(doc_topic_dists, columns=[f"Topic {i+1}" for i in range(n_topics)])
print("\nDocument-Topic Distributions:")
print(doc_topic_df.round(3).to_string(index=False))

# Step 7: Visualize Top Words per Topic (Bar Charts)
# This function creates and saves a bar chart for each topic,
# showing the top words and how important they are to that topic.
def plot_top_words_per_topic(model, feature_names, n_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        top_weights = topic[top_indices]

        plt.figure(figsize=(8, 4))
        sns.barplot(x=top_weights, y=top_words)
        plt.title(f"Topic #{topic_idx + 1} - Top Words")
        plt.xlabel("Word Importance")
        plt.tight_layout()
        plt.savefig(f"topic_{topic_idx + 1}_top_words.png")
        plt.close()
        print(f"Saved: topic_{topic_idx + 1}_top_words.png")

plot_top_words_per_topic(lda, feature_names)

# Step 8: Visualize Document-Topic Heatmap
# This heatmap shows the topic distribution across documents.
# Each cell indicates how strongly a topic is represented in a document.
plt.figure(figsize=(8, 4))
sns.heatmap(doc_topic_df, annot=True, cmap="YlGnBu", cbar=True)
plt.title("Document-Topic Distribution Heatmap")
plt.xlabel("Topics")
plt.ylabel("Documents")
plt.tight_layout()
plt.savefig("document_topic_heatmap.png")
plt.close()
print("Saved: document_topic_heatmap.png")
