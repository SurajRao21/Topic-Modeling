# Sentiment Association Networks

# Run:
# !pip install nltk pandas matplotlib numpy networkx scikit-learn


import nltk
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import re
from collections import Counter, defaultdict
from itertools import combinations
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ----------------------------
# 0) NLTK downloads
# ----------------------------
nltk.download("punkt")
nltk.download("stopwords")
try:
    nltk.download("punkt_tab")  # harmless if not available
except Exception:
    pass
try:
    nltk.download("vader_lexicon")
except Exception:
    pass

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# ----------------------------------------------------
# 1) Create a mixed dataset (50 small documents)
# ----------------------------------------------------
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
# 2) Preprocess & VADER sentiment
# ----------------------------------------------------
stop_words = set(stopwords.words("english")) | set(ENGLISH_STOP_WORDS)

def tokenize_clean(text: str):
    # lower, tokenize, keep alphabetic terms, remove stopwords
    toks = word_tokenize(text.lower())
    return [w for w in toks if w.isalpha() and w not in stop_words]

tokenized_docs = [tokenize_clean(doc) for doc in documents]

# Sentiment scoring (per document)
sid = SentimentIntensityAnalyzer()
scores = [sid.polarity_scores(doc) for doc in documents]
df = pd.DataFrame(scores)
df["text"] = documents

# Label using a neutral band (VADER convention: +/- 0.05)
def label_from_compound(c, pos_thr=0.05, neg_thr=-0.05):
    if c >= pos_thr:
        return "pos"
    if c <= neg_thr:
        return "neg"
    return "neu"

df["label"] = df["compound"].apply(label_from_compound)

# ----------------------------------------------------
# 3) Quick overview plots
# ----------------------------------------------------
# (A) Distribution of sentiment classes
plt.figure(figsize=(6, 4))
df["label"].value_counts().reindex(["neg","neu","pos"]).fillna(0).plot(kind="bar")
plt.title("Sentiment Class Distribution")
plt.xlabel("Class"); plt.ylabel("Count"); plt.tight_layout()
plt.show()

# (B) Compound score histogram
plt.figure(figsize=(6, 4))
plt.hist(df["compound"], bins=10)
plt.title("Compound Score Distribution")
plt.xlabel("Compound (-1 to 1)"); plt.ylabel("Frequency"); plt.tight_layout()
plt.show()

# ----------------------------------------------------
# 4) Term ↔ Sentiment Association via PMI
# ----------------------------------------------------
# Build binary presence matrices per doc for terms and for sentiment class.
# Then compute PMI(term, class) = log( P(t,c) / (P(t)*P(c)) ), with Laplace smoothing.
docs_terms = [set(toks) for toks in tokenized_docs]
all_terms = sorted(set().union(*docs_terms))
classes = ["neg","neu","pos"]
N = len(documents)

# Counts
term_counts = Counter()
class_counts = Counter(df["label"])
joint_counts = {c: Counter() for c in classes}

for i, terms in enumerate(docs_terms):
    c = df.loc[i, "label"]
    for t in terms:
        term_counts[t] += 1
        joint_counts[c][t] += 1

# PMI with Laplace smoothing
def pmi(term, cls, alpha=1.0):
    ct = term_counts[term]
    cc = class_counts[cls]
    ctc = joint_counts[cls][term]

    # smoothed probabilities
    p_t = (ct + alpha) / (N + alpha * len(all_terms))
    p_c = (cc + alpha) / (N + alpha * len(classes))
    p_tc = (ctc + alpha) / (N + alpha * (len(all_terms) * len(classes)))

    return np.log(p_tc / (p_t * p_c))

# Build PMI matrix (terms x classes)
pmi_data = []
for t in all_terms:
    row = {"term": t}
    for c in classes:
        row[c] = pmi(t, c)
    pmi_data.append(row)

pmi_df = pd.DataFrame(pmi_data).set_index("term")

# ----------------------------------------------------
# 5) Heatmap of top terms by overall association strength
# ----------------------------------------------------
# Score terms by max absolute PMI across classes (strongest association wins)
pmi_df["max_abs"] = pmi_df[classes].abs().max(axis=1)
TOP_N = 20  # adjust to show more/less terms
top_terms = pmi_df.sort_values("max_abs", ascending=False).head(TOP_N).index

matrix_top = pmi_df.loc[top_terms, classes]

plt.figure(figsize=(10, 8))
im = plt.imshow(matrix_top.values, aspect="auto", interpolation="nearest", cmap="coolwarm")
plt.colorbar(im, fraction=0.046, pad=0.04, label="PMI (term ↔ sentiment)")
plt.title(f"Term × Sentiment Association (PMI) — Top {TOP_N} terms", fontsize=14)
plt.xticks(ticks=np.arange(len(classes)), labels=classes)
plt.yticks(ticks=np.arange(len(top_terms)), labels=top_terms)
# add cell labels
for i in range(matrix_top.shape[0]):
    for j in range(matrix_top.shape[1]):
        val = matrix_top.values[i, j]
        plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
plt.tight_layout()
plt.show()

# ----------------------------------------------------
# 6) Bipartite Association Network (terms ↔ sentiment)
# ----------------------------------------------------
# Connect each top term to the *single* sentiment class with which it has
# the highest positive PMI. Threshold to declutter.
G = nx.Graph()
term_nodes = []
sent_nodes = classes  # "neg","neu","pos"

PMI_MIN = 0.2  # edges must exceed this PMI to be drawn

for t in top_terms:
    best_cls = matrix_top.loc[t].idxmax()
    best_val = matrix_top.loc[t, best_cls]
    if best_val > PMI_MIN:
        G.add_node(t, bipartite="term")
        G.add_node(best_cls, bipartite="sentiment")
        G.add_edge(t, best_cls, weight=float(best_val))
        term_nodes.append(t)

plt.figure(figsize=(12, 9))
# layout: fix sentiment nodes on one side, terms on the other (bipartite layout)
pos = {}
# sentiment on left
for i, s in enumerate(sent_nodes):
    pos[s] = (-1.5, 1 - i)
# terms spread on right
for i, t in enumerate(term_nodes):
    pos[t] = (0.8 + (i % 4) * 0.3, 1 - (i // 4) * 0.4)

edges = G.edges(data=True)
weights = [d["weight"] for (_, _, d) in edges]
nx.draw_networkx_nodes(G, pos, nodelist=sent_nodes, node_color="lightgray", node_size=1200, alpha=0.9)
nx.draw_networkx_nodes(G, pos, nodelist=term_nodes, node_color="skyblue", node_size=700, alpha=0.9)
nx.draw_networkx_edges(G, pos, width=[max(1.0, w) for w in weights], alpha=0.7)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

plt.title("Term ↔ Sentiment Association Network (PMI)", fontsize=16)
plt.axis("off")
plt.tight_layout()
plt.show()

# ----------------------------------------------------
# 7) Save artifacts
# ----------------------------------------------------
# Full per-document sentiment
df[["text", "neg", "neu", "pos", "compound", "label"]].to_csv("corpus_with_sentiment.csv", index=False, encoding="utf-8")

# PMI matrix for all terms
pmi_df[classes].to_csv("term_sentiment_pmi.csv", encoding="utf-8")

# Top-terms heatmap data
matrix_top.to_csv("term_sentiment_pmi_top.csv", encoding="utf-8")

# Save last figure (heatmap) as PNG as well (re-plot quickly just to save cleanly)
plt.figure(figsize=(10, 8))
im = plt.imshow(matrix_top.values, aspect="auto", interpolation="nearest", cmap="coolwarm")
plt.colorbar(im, fraction=0.046, pad=0.04, label="PMI (term ↔ sentiment)")
plt.title(f"Term × Sentiment Association (PMI) — Top {TOP_N} terms", fontsize=14)
plt.xticks(ticks=np.arange(len(classes)), labels=classes)
plt.yticks(ticks=np.arange(len(top_terms)), labels=top_terms)
for i in range(matrix_top.shape[0]):
    for j in range(matrix_top.shape[1]):
        val = matrix_top.values[i, j]
        plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
plt.tight_layout()
plt.savefig("term_sentiment_pmi_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved:")
print(" - corpus_with_sentiment.csv")
print(" - term_sentiment_pmi.csv")
print(" - term_sentiment_pmi_top.csv")
print(" - term_sentiment_pmi_heatmap.png")
