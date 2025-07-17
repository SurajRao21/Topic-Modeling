# Step 1: Import Libraries
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

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

