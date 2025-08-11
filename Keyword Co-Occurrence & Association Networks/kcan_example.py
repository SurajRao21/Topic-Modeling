# Keyword Co-Occurrence & Association Networks 


import nltk
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import Counter
from itertools import combinations

# -------------------------------------------------------------------
# 0) Download required NLTK data (tokenizer + stopwords)
# -------------------------------------------------------------------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
# Some newer NLTK installs also provide punkt_tab; it's safe to try.
try:
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ----------------------------------------------------
# 1) Create a mixed dataset (50 small documents)
# ----------------------------------------------------
# You can replace this list with your own corpus.
documents = [
    # Environment-related
    "Climate change impacts biodiversity across the globe.",
    "Deforestation contributes to rising carbon dioxide levels.",
    "Renewable energy sources like wind and solar are crucial.",
    "Ocean pollution is harming marine life ecosystems.",
    "Sustainable farming can reduce environmental damage.",
    "Wildlife conservation helps preserve endangered species.",
    "Plastic waste management is a pressing global issue.",
    "Air quality in urban areas is deteriorating due to emissions.",
    "Water scarcity affects millions worldwide.",
    "Global warming increases the risk of natural disasters.",

    # Technology-related
    "Artificial intelligence is transforming industries.",
    "Quantum computing will revolutionize cryptography.",
    "5G networks enable faster communication speeds.",
    "Cybersecurity is critical in the digital age.",
    "Machine learning models improve with more data.",
    "Blockchain technology supports decentralized finance.",
    "Augmented reality is changing retail experiences.",
    "Cloud computing offers scalable storage solutions.",
    "Autonomous vehicles rely on advanced sensors.",
    "Robotics is enhancing manufacturing efficiency.",

    # Mixed general topics
    "Electric vehicles reduce dependency on fossil fuels.",
    "Smart grids optimize renewable energy distribution.",
    "Green data centers improve energy efficiency.",
    "Biotechnology innovations support medical research.",
    "Urban planning integrates green infrastructure.",
    "Space exploration expands human knowledge.",
    "Satellite imagery monitors environmental changes.",
    "Digital twins simulate real-world systems.",
    "Nanotechnology advances material science.",
    "Wearable devices track health metrics.",

    # More mixed/overlap topics
    "AI helps optimize solar panel performance.",
    "IoT sensors track air pollution in real time.",
    "Drones assist in wildlife monitoring.",
    "Energy storage batteries support renewable power.",
    "Precision agriculture uses satellite and AI data.",
    "Marine robots explore ocean ecosystems.",
    "Smart homes use energy efficiently.",
    "3D printing creates sustainable building materials.",
    "Big data analytics inform climate policy.",
    "Robotics assists in disaster recovery.",

    # Random extra tech/env examples
    "Hydroelectric dams generate clean energy.",
    "Digital currencies disrupt financial systems.",
    "Recycling technologies improve waste processing.",
    "AI-driven healthcare diagnostics save lives.",
    "Cloud platforms support remote workforces.",
    "Wildfires are becoming more frequent worldwide.",
    "Advanced sensors detect methane leaks.",
    "Self-driving trucks deliver goods autonomously.",
    "Satellite networks provide internet to rural areas.",
    "Urban green spaces improve mental health."
]

# ----------------------------------------------------
# 2) Preprocess: tokenize, lowercase, remove stopwords
# ----------------------------------------------------
stop_words = set(stopwords.words('english'))

tokenized_docs = []
for doc in documents:
    # word_tokenize splits text into tokens; lower() for case-insensitivity
    tokens = word_tokenize(doc.lower())
    # keep alphabetic tokens and drop stopwords to focus on keywords
    filtered_tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    tokenized_docs.append(filtered_tokens)

# ----------------------------------------------------
# 3) Count co-occurrences of keywords
# ----------------------------------------------------
# We count pairwise co-occurrence *within the same document*.
# Using set(tokens) avoids counting the same pair multiple times per doc.
co_occurrences = Counter()
for tokens in tokenized_docs:
    for pair in combinations(set(tokens), 2):
        co_occurrences[tuple(sorted(pair))] += 1

# ----------------------------------------------------
# 4) Build the network graph
# ----------------------------------------------------
# Edges connect words; edge weights = number of documents where the pair co-occurs.
G = nx.Graph()
for (word1, word2), count in co_occurrences.items():
    if count >= 2:  # threshold to declutter the graph; adjust as needed
        G.add_edge(word1, word2, weight=count)

# ----------------------------------------------------
# 5) Visualize the co-occurrence network
# ----------------------------------------------------
plt.figure(figsize=(15, 12))
pos = nx.spring_layout(G, k=0.5, seed=42)  # force-directed layout

edges = G.edges(data=True)
weights = [d['weight'] for (_, _, d) in edges]

# Draw nodes/edges/labels; edge width encodes co-occurrence strength
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue', alpha=0.9)
nx.draw_networkx_edges(G, pos, width=[w * 0.5 for w in weights], alpha=0.6)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

plt.title("Keyword Co-Occurrence & Association Network", fontsize=16)
plt.axis("off")
plt.show()

# ----------------------------------------------------
# 6) Heatmap of co-occurrence strengths (Top N terms + value labels)
# ----------------------------------------------------
# Build a symmetric co-occurrence matrix (adjacency-style) for all terms
terms = sorted({t for pair in co_occurrences for t in pair})
matrix = pd.DataFrame(0, index=terms, columns=terms, dtype=int)

for (w1, w2), count in co_occurrences.items():
    matrix.at[w1, w2] = count
    matrix.at[w2, w1] = count

# Rank terms by total co-occurrence strength and keep top N
TOP_N = 20  # <-- adjust this to show more/less terms in the heatmap
term_strength = matrix.sum(axis=1).sort_values(ascending=False)
top_terms = term_strength.head(TOP_N).index.tolist()

# Subset and reorder by strength for readability
matrix_top = matrix.loc[top_terms, top_terms]
order = matrix_top.sum(axis=1).sort_values(ascending=False).index.tolist()
matrix_top = matrix_top.loc[order, order]

# Plot heatmap with values using matplotlib (no seaborn needed)
plt.figure(figsize=(12, 10))
im = plt.imshow(matrix_top.values, aspect='auto', interpolation='nearest', cmap='YlGnBu')
cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.set_label('Co-occurrence count')

# Add numeric value labels in each cell
for i in range(matrix_top.shape[0]):
    for j in range(matrix_top.shape[1]):
        value = int(matrix_top.values[i, j])
        plt.text(j, i, str(value), ha='center', va='center', color='black', fontsize=8)

plt.title(f"Keyword Co-occurrence Heatmap (Top {TOP_N} terms)", fontsize=16)
plt.xticks(ticks=np.arange(len(matrix_top.columns)), labels=matrix_top.columns, rotation=90)
plt.yticks(ticks=np.arange(len(matrix_top.index)), labels=matrix_top.index)
plt.tight_layout()
plt.show()

# ----------------------------
# Save artifacts
# ----------------------------
matrix_top.to_csv("cooccurrence_matrix.csv", encoding="utf-8", index=True)
plt.savefig("cooccurrence_heatmap.png", dpi=150, bbox_inches="tight")
