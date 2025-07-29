# Load necessary libraries
library(lsa)
library(quanteda)
library(ggplot2)

# Define the documents (Example corpus)
documents <- c(
  "TESLA!!! has announced its *NEW* electric car model...which is expected to 100% revolutionize the auto industry.",
  "The stock-market experienced a significant DROP, due to recent ECONOMIC uncertainty?!",
  "Scientists have discovered a *NEW* species of dino...saur??? in the desert-region of North America.",
  "Tech-industry continues to grow...RAPIDLY, with new startups emerging... especially in #AI & machine-learning.",
  "NASA...successfully launched 1 new satellite: aimed at exploring distant planets--awesome!"
)

# Step 1: Text Preprocessing using quanteda
corpus <- corpus(documents)
tokens <- tokens(corpus, remove_punct = TRUE, remove_numbers = TRUE, remove_symbols = TRUE)
tokens <- tokens_tolower(tokens)
tokens <- tokens_remove(tokens, stopwords("english"))

# Create a Document-Term Matrix (DTM) using quanteda
dtm <- dfm(tokens)

# Convert the DTM to a regular matrix (as required by lsa package)
dtm_matrix <- as.matrix(dtm)

# Step 2: Apply LSA using Singular Value Decomposition (SVD)
lsa_space <- lsa(dtm_matrix)

# Step 3: Get the term names (words) from the original DTM
term_names <- colnames(dtm_matrix)

# Step 4: Extract the first 2 topics (concepts) from the LSA space
k <- 2  # Number of topics you want
lsa_topics <- lsa_space$tk[, 1:k]  # Extract first 'k' topics (term-topic matrix)

# Step 5: Get top terms for each topic (by sorting the values of each topic)
get_top_terms <- function(lsa_space, term_names, k = 2, n_terms = 5) {
  term_topic_matrix <- lsa_space$tk
  top_terms <- list()
  
  for (i in 1:k) {
    top_indices <- order(term_topic_matrix[, i], decreasing = TRUE)[1:n_terms]
    top_terms[[paste("Topic", i)]] <- term_names[top_indices]
  }
  
  return(top_terms)
}

# Get the top terms for each topic
top_terms_lsa <- get_top_terms(lsa_space, term_names, k = 2, n_terms = 5)

# Display the results
cat("\nTop 5 terms for each LSA topic:\n")
print(top_terms_lsa)

# Step 6: Visualize the terms for each topic

# Create a data frame for the top terms for visualization
top_terms_df <- data.frame(
  Term = c(top_terms_lsa$`Topic 1`, top_terms_lsa$`Topic 2`),
  Topic = rep(c("Topic 1", "Topic 2"), each = 5),
  Importance = c(lsa_space$tk[order(lsa_space$tk[,1], decreasing = TRUE)[1:5], 1], 
                 lsa_space$tk[order(lsa_space$tk[,2], decreasing = TRUE)[1:5], 2])
)

# Visualization of the terms for each topic
ggplot(top_terms_df, aes(x = Term, y = Importance, fill = Topic)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Top Terms for Each LSA Topic", x = "Terms", y = "Importance") +
  theme_minimal()

