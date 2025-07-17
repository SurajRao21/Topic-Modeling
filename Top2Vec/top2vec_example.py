# Step 1: Import Libraries
import pandas as pd
from top2vec import Top2Vec

# Step 2: Load and filter relevant summaries
df = pd.read_csv("summaries.csv")  # Ensure the CSV file path is correct
relevant_summaries = df[df["label"] == 1]["summary"].tolist()  # Only consider summaries with label = 1

# Step 3: Fit Top2Vec Model
# Initialize and fit Top2Vec with the relevant summaries
model = Top2Vec(relevant_summaries, min_count=2, workers=4)  # min_count=2 ensures only topics with at least 2 docs are considered

# Step 4: Get Topics
# Get the most relevant words for each topic
topic_words, word_scores, topic_nums = model.get_topics()

# Step 5: Print the top words for each topic
print("Identified Topics:")
for i, topic in enumerate(topic_words, 1):
    print(f"Topic #{i}: {', '.join(topic)}")