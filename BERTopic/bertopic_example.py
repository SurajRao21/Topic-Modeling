# Step 1: Import Libraries
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora
from gensim.models import CoherenceModel

# Step 2: Load and Filter Relevant Summaries
# Load the dataset (make sure the 'summaries.csv' file is in the correct path)
df = pd.read_csv("summaries.csv")

# Filter relevant summaries (assuming label == 1 is the relevant summary)
relevant_summaries = df[df["label"] == 1]["summary"].tolist()

# Step 3: Configure BERTopic to Extract Many Small Topics
# Here we use CountVectorizer to process the text and remove common English stopwords
vectorizer_model = CountVectorizer(stop_words="english")

# Initialize BERTopic model with a lower threshold for topic size (min_topic_size=2) to extract many smaller topics
topic_model = BERTopic(language="english", vectorizer_model=vectorizer_model, min_topic_size=2)

# Step 4: Fit BERTopic Model
# The fit_transform method takes the relevant summaries and fits the model, 
# returning the topics and their associated probabilities (we'll ignore the probabilities here)
topics, _ = topic_model.fit_transform(relevant_summaries)

# Step 5: Extract and Print Unique Topic Names
# Extract topic information to get the topic names
topic_info = topic_model.get_topic_info()

# Filter out any unassigned topics (-1) and get the unique topic names
unique_topics = topic_info[topic_info.Topic != -1]["Name"].unique().tolist()

# Check if any valid topics were found and print them
if not unique_topics:
    print("No valid topics were found.")
else:
    print("Identified Topics:")
    for i, topic in enumerate(unique_topics, 1):
        print(f"{i}. {topic}")

# Step 6: Extract Top Words for Each Topic
# Extract the top 10 words for each topic (you can adjust this number)
top_words_per_topic = topic_model.get_topics()

# Step 7: Prepare Gensim Dictionary and Corpus
# Prepare the corpus and dictionary from the relevant summaries
texts = [summary.split() for summary in relevant_summaries]  # Tokenize summaries
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Step 8: Calculate Coherence Score using Gensim's CoherenceModel
# We need to pass the top words (as a list of words) and the dictionary for coherence calculation
topics_for_coherence = []

# Iterate over the extracted topics and add the top words to topics_for_coherence
for i in range(len(top_words_per_topic)):
    # Check if the topic index exists (i.e., is not missing or out of range)
    if i in top_words_per_topic:
        topics_for_coherence.append([word[0] for word in top_words_per_topic[i]])

# Calculate coherence score
coherence_model = CoherenceModel(topics=topics_for_coherence, texts=texts, dictionary=dictionary, coherence='c_w2v')
coherence_score = coherence_model.get_coherence()

print(f"Coherence Score: {coherence_score}")
