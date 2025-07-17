# Step 1: Install and load necessary packages
install.packages("stm")
install.packages("tidyverse")
install.packages("tm")  # for text processing

library(stm)
library(tidyverse)
library(tm)

# Step 2: Example text data (Increase the number of documents for better results)
texts <- c(
  "Topic modeling is an interesting approach to extracting topics from text data.",
  "This approach works well for news articles and other types of textual data.",
  "Data science is a growing field with many applications in technology.",
  "Text mining techniques are valuable for understanding patterns in text.",
  "The relationship between language and meaning is complex and multi-faceted.",
  "The rise of artificial intelligence is transforming many industries.",
  "Machine learning algorithms are crucial in data science.",
  "Understanding text patterns can reveal a lot of insights from large datasets.",
  "Artificial intelligence is used to solve complex problems in many fields.",
  "Data mining and topic modeling are methods used in text analysis."
)

# Step 3: Create metadata (Example metadata for documents)
metadata <- data.frame(doc_id = 1:length(texts),
                       document_type = c("Research", "News", "Research", "Tech", "AI", "Tech", "AI", "Research", "Tech", "AI"))

# Step 4: Preprocess the text data using the `tm` package
corpus <- Corpus(VectorSource(texts))
corpus <- tm_map(corpus, content_transformer(tolower))  # Convert to lowercase
corpus <- tm_map(corpus, removePunctuation)  # Remove punctuation
corpus <- tm_map(corpus, removeNumbers)  # Remove numbers
corpus <- tm_map(corpus, removeWords, stopwords("en"))  # Remove common stopwords
corpus <- tm_map(corpus, stripWhitespace)  # Remove extra whitespace

# Convert the cleaned corpus to a Document-Term Matrix (DTM)
dtm <- DocumentTermMatrix(corpus)

# Step 5: Prepare documents for STM
processed <- textProcessor(texts, metadata = metadata)
out <- prepDocuments(processed$documents, processed$vocab, processed$meta)

# Get the documents, vocabulary, and metadata from the output
docs <- out$documents  # Document-term matrix
vocab <- out$vocab  # Vocabulary
meta <- out$meta  # Metadata

# Step 6: Fit the STM model (K = 2 topics for a small dataset)
stm_model <- stm(docs, vocab, K = 2, max.em.its = 75, data = meta, init.type = "Spectral")

# Step 7: View the summary of the STM model
summary(stm_model)

# Step 8: Examine the topics - top words for each topic
labelTopics(stm_model)

# Step 9: Visualize the Topics

# Plot topic prevalence (how each topic is distributed across documents)
plot(stm_model, type = "summary")

# Plot topic-word distribution (which words are most relevant for each topic)
# Specify topics to visualize, e.g., topics 1 and 2
plot(stm_model, type = "perspectives", topics = c(1, 2))

