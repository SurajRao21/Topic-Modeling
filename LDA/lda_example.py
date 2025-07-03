#LDA (Latent Dirichlet Allocation)
#LDA is a generative model that treats documents as mixtures of topics and topics as mixtures of words. Each document is represented as a probability distribution over topics.

# Uncomment the following line to install the required libraries for the first time
# !pip install scikit-learn top2vec bertopic pandas

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Sample data
documents = [
    "Tesla has announced its new electric car model, which is expected to revolutionize the auto industry.",
    "The stock market experienced a significant drop due to recent economic uncertainty.",
    "Scientists have discovered a new species of dinosaur in the desert region of North America.",
    "The tech industry continues to grow rapidly, with new startups emerging in the AI space.",
    "NASA successfully launched a new satellite aimed at exploring distant planets."
]

# Vectorize the text data
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# Apply LDA
lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(X)

# Display topics
for topic_idx, topic in enumerate(lda.components_):
    print(f"Topic #{topic_idx}:")
    print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-10 - 1:-1]])
