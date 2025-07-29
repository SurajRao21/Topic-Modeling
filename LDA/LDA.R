# Load necessary libraries
library(topicmodels)
library(tm)
library(ggplot2)

# Define the documents (Example corpus)
documents <- c(
  "TESLA!!! has announced its *NEW* electric car model...which is expected to 100% revolutionize the auto industry.",
  "The stock-market experienced a significant DROP, due to recent ECONOMIC uncertainty?!",
  "Scientists have discovered a *NEW* species of dino...saur??? in the desert-region of North America.",
  "Tech-industry continues to grow...RAPIDLY, with new startups emerging... especially in #AI & machine-learning.",
  "NASA...successfully launched 1 new satellite: aimed at exploring distant planets--awesome!"
)

# Step 1: Text Preprocessing using tm
corpus <- Corpus(VectorSource(documents))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stripWhitespace)

# Create a Document-Term Matrix (DTM) using tm
dtm <- DocumentTermMatrix(corpus)

# Step 2: Apply LDA using the topicmodels package
k <- 2  # Set the number of topics to 2
lda_model <- LDA(dtm, k = k, control = list(seed = 1234))

# Step 3: Get the top 5 terms for each topic
top_terms_lda <- terms(lda_model, 5)  # Get the top 5 terms for each topic

# Display the top 5 terms for each topic
cat("\nTop 5 terms for each LDA topic:\n")
print(top_terms_lda)

# Step 4: Convert the top terms into a data frame for visualization
lda_terms_df <- data.frame(
  Term = c(top_terms_lda[, 1], top_terms_lda[, 2]),  # Extract terms from both topics
  Topic = rep(c("Topic 1", "Topic 2"), each = 5),  # Assign each term to its topic
  Importance = rep(1, 10)  # All terms have equal importance for simplicity
)

# Step 5: Visualization of the terms for each topic
ggplot(lda_terms_df, aes(x = Term, y = Importance, fill = Topic)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Top Terms for Each LDA Topic", x = "Terms", y = "Importance") +
  theme_minimal()

