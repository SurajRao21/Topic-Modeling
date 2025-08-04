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
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
import multiprocessing
from multiprocessing import freeze_support

# Step 0: Download necessary NLTK resources
# These are required for text preprocessing: lemmatization and stopword removal.
nltk.download('wordnet')
nltk.download('stopwords')

def main():
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
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(cleaned_documents)
    feature_names = vectorizer.get_feature_names_out()

    # Step 4: Fit LDA Topic Model
    n_topics = 2
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)

    # Step 5: Print Top Words for Each Topic
    n_top_words = 10
    for topic_idx, topic in enumerate(lda.components_):
        print(f"\nTopic #{topic_idx + 1}:")
        top_features = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        print(", ".join(top_features))

    # Step 6: Document-Topic Distribution
    doc_topic_dists = lda.transform(X)
    doc_topic_df = pd.DataFrame(doc_topic_dists, columns=[f"Topic {i+1}" for i in range(n_topics)])
    print("\nDocument-Topic Distributions:")
    print(doc_topic_df.round(3).to_string(index=False))

    # Step 7: Topic Coherence Calculation (using C_UCI)
    topics = []
    for topic_weights in lda.components_:
        top_indices = topic_weights.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        topics.append(top_words)

    # Create a Gensim Dictionary and Corpus
    tokenized_docs = [doc.split() for doc in cleaned_documents]
    dictionary = Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(text) for text in tokenized_docs]

    # Now, calculate the coherence score using the top words from LDA model
    coherence_model_lda = CoherenceModel(topics=topics, texts=tokenized_docs, dictionary=dictionary, coherence='c_uci')
    coherence_score = coherence_model_lda.get_coherence()
    print(f"\nLDA Model C_UCI Coherence Score: {coherence_score:.4f}")

    # Step 8: Visualize Top Words per Topic (Bar Charts)
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
            plt.show()
            plt.close()
            print(f"Saved: topic_{topic_idx + 1}_top_words.png")

    plot_top_words_per_topic(lda, feature_names)

    # Step 9: Visualize Document-Topic Heatmap
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


if __name__ == '__main__':
    freeze_support()  # for Windows-based systems
    main()
