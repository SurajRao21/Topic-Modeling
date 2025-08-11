import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Bigger, mixed dataset
text = """
Although climate change is a global issue, some countries invest little in renewable energy.
Many urban areas face air pollution problems because of traffic congestion.
However, electric vehicles are becoming more popular in cities.
Then governments started offering subsidies for solar panels.
Wildlife populations are declining since deforestation continues at a rapid pace.
In addition, ocean plastic pollution threatens marine ecosystems.
The tech industry is booming, but cybersecurity risks are growing.
AI is transforming healthcare because it can detect diseases early.
Moreover, cloud computing makes data storage more efficient.
Yet, some rural areas still lack reliable internet connections.
After the pandemic, remote work became common for many industries.
"""

# Discourse marker categories
DISCOURSE_MARKERS = {
    "contrast": ["although", "however", "but", "yet", "though", "nevertheless"],
    "cause": ["because", "since", "as", "therefore", "thus", "hence", "so"],
    "temporal": ["then", "after", "before", "meanwhile", "eventually", "finally"],
    "addition": ["also", "moreover", "furthermore", "in addition"],
}

# Flatten for quick lookup
marker_to_category = {}
for cat, words in DISCOURSE_MARKERS.items():
    for w in words:
        marker_to_category[w] = cat

doc = nlp(text)
marker_counts = {cat: 0 for cat in DISCOURSE_MARKERS}

print("=== Sentence-level discourse relations ===")
for sent in doc.sents:
    tokens = list(sent)
    found = [tok for tok in tokens if tok.text.lower() in marker_to_category]
    if not found:
        print(f"- {sent.text.strip()} | markers: none")
        continue
    for tok in found:
        m = tok.text.lower()
        cat = marker_to_category[m]
        marker_counts[cat] += 1
        left_clause = sent.text[: tok.idx - sent.start_char].strip(",;:-— ").strip()
        right_clause = sent.text[tok.idx - sent.start_char + len(tok):].strip(",;:-— ").strip()
        if not left_clause:
            left_clause = "(preceding clause)"
        if not right_clause:
            right_clause = "(following clause)"
        print(f"- {cat.upper():9s} | {left_clause} —[{tok.text}]→ {right_clause}")

print("\n=== Marker counts by category ===")
for cat, c in marker_counts.items():
    print(f"{cat:9s}: {c}")
