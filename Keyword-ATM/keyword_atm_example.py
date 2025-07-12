# Required installations (if not already installed):
# !pip install nltk scikit-learn pandas numpy matplotlib seaborn

import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

# Step 0: Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

# Step 1: Input Documents
documents = [
    "TESLA!!! has announced its *NEW* electric car model...which is expected to 100% revolutionize the auto industry.",
    "The stock-market experienced a significant DROP, due to recent ECONOMIC uncertainty?!",
    "Scientists have discovered a *NEW* species of dino...saur??? in the desert-region of North America.",
    "Tech-industry continues to grow...RAPIDLY, with new startups emerging... especially in #AI & machine-learning.",
    "NASA...successfully launched 1 new satellite: aimed at exploring distant planets--awesome!"
]

# Step 2: Preprocessing Function 
def clean_text(doc):
    doc = doc.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(doc)
    cleaned = [token for token in tokens if token not in stopwords.words('english') and len(token) > 2]
    return cleaned

# Apply cleaning to all documents
cleaned_documents = [clean_text(doc) for doc in documents]

# Step 3: Text Vectorization 
vectorizer = CountVectorizer(analyzer=lambda x: x)  # Passing tokenized words directly
X = vectorizer.fit_transform(cleaned_documents)
feature_names = vectorizer.get_feature_names_out()

# Step 4: Keyword Assistance (Specify Keywords)
# Define keywords that you want the model to focus on
keywords = ["electric", "stock", "AI", "satellite", "tech"]

# Step 5: Fit LDA Model with Keywords Assistance
n_topics = 2  # Number of topics
lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)

# Fit the model on the document-term matrix
lda_model.fit(X)

# Step 6: Print Top Words for Each Topic
n_top_words = 10
for topic_idx, topic in enumerate(lda_model.components_):
    top_features = np.array(feature_names)[np.argsort(topic)[::-1][:n_top_words]]
    print(f"\nTopic #{topic_idx + 1}:")
    print(", ".join(top_features))

# Step 7: Visualize Top Words per Topic (Bar Charts)
def plot_top_words_per_topic(model, feature_names, n_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        top_indices = np.argsort(topic)[::-1][:n_top_words]
        top_words = np.array(feature_names)[top_indices]
        top_weights = topic[top_indices]

        plt.figure(figsize=(8, 4))
        sns.barplot(x=top_weights, y=top_words)
        plt.title(f"Topic #{topic_idx + 1} - Top Words")
        plt.xlabel("Word Importance")
        plt.tight_layout()
        plt.savefig(f"topic_{topic_idx + 1}_top_words.png")
        plt.show()
        plt.close()
        print(f"Saved: topic_{topic_idx + 1}_top_words.png")

plot_top_words_per_topic(lda_model, feature_names)

# Step 8: Visualize Document-Topic Heatmap
doc_topic_dists = lda_model.transform(X)  # Get topic distribution for each document
doc_topic_df = pd.DataFrame(doc_topic_dists, columns=[f"Topic {i+1}" for i in range(n_topics)])

# Display the heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(doc_topic_df, annot=True, cmap="YlGnBu", cbar=True)
plt.title("Document-Topic Distribution Heatmap")
plt.xlabel("Topics")
plt.ylabel("Documents")
plt.tight_layout()
plt.savefig("document_topic_heatmap.png")
plt.show()
plt.close()
print("Saved: document_topic_heatmap.png")
