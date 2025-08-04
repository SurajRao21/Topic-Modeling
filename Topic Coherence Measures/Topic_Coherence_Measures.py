import pandas as pd
import numpy as np
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
import matplotlib.pyplot as plt

# Sample documents - mix of different domains
sample_documents = [
    # Technology documents
    "Machine learning algorithms are revolutionizing data science and artificial intelligence applications",
    "Deep learning neural networks require massive computational power and GPU acceleration",
    "Python programming language is widely used for data analysis and machine learning projects",
    "Cloud computing platforms provide scalable infrastructure for big data processing",
    "Software development teams use version control systems like Git for collaboration",
    "Database management systems store and retrieve large amounts of structured data efficiently",
    "Cybersecurity threats require advanced encryption and network security protocols",
    
    # Healthcare documents  
    "Medical diagnosis relies on patient symptoms and comprehensive clinical examinations",
    "Hospital emergency departments treat patients with urgent medical conditions",
    "Pharmaceutical research develops new drugs and therapeutic treatments for diseases",
    "Health insurance coverage affects patient access to medical care and treatments",
    "Nurses provide direct patient care and monitor vital signs in hospital settings",
    "Medical imaging techniques like MRI and CT scans help diagnose internal conditions",
    "Preventive healthcare focuses on wellness programs and early disease detection",
    
    # Finance documents
    "Stock market investments require careful analysis of financial performance and market trends",
    "Banking institutions provide loans and credit services to individual and business customers",
    "Cryptocurrency trading involves digital assets and blockchain technology platforms",
    "Insurance policies protect against financial losses from accidents and disasters",
    "Retirement planning requires long-term investment strategies and portfolio diversification",
    "Corporate finance teams manage company budgets and capital allocation decisions",
    "Economic indicators influence monetary policy and interest rate decisions",
    
    # Mixed/General documents
    "Online education platforms offer courses in technology, business, and healthcare fields",
    "Remote work arrangements require reliable internet connectivity and collaboration tools",
    "Consumer behavior analysis helps companies understand market demands and preferences",
    "Environmental sustainability initiatives focus on renewable energy and waste reduction"
]

def preprocess_documents(documents):
    """
    Clean and preprocess documents for topic modeling
    """
    processed_docs = []
    
    for doc in documents:
        # Convert to lowercase
        doc = doc.lower()
        
        # Remove punctuation and numbers
        doc = re.sub(r'[^a-zA-Z\s]', '', doc)
        
        # Split into words
        words = doc.split()
        
        # Remove stop words and short words
        words = [word for word in words if word not in ENGLISH_STOP_WORDS and len(word) > 3]
        
        processed_docs.append(words)
    
    return processed_docs

def create_lda_model(processed_docs, num_topics=3):
    """
    Create LDA topic model
    """
    # Create dictionary and corpus
    dictionary = corpora.Dictionary(processed_docs)
    
    # Filter extreme cases
    dictionary.filter_extremes(no_below=2, no_above=0.7)
    
    # Create corpus (bag of words)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    
    # Train LDA model
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )
    
    return lda_model, dictionary, corpus

def evaluate_coherence(model, texts, dictionary, corpus):
    """
    Evaluate topic coherence using different measures
    """
    coherence_measures = {}
    
    # C_v coherence (most common, correlates well with human judgment)
    coherence_cv = CoherenceModel(
        model=model, 
        texts=texts, 
        dictionary=dictionary, 
        coherence='c_v'
    )
    coherence_measures['C_v'] = coherence_cv.get_coherence()
    
    # UMass coherence (intrinsic measure, uses corpus)
    coherence_umass = CoherenceModel(
        model=model, 
        corpus=corpus, 
        dictionary=dictionary, 
        coherence='u_mass'
    )
    coherence_measures['UMass'] = coherence_umass.get_coherence()
    
    # C_uci coherence (based on PMI)
    coherence_uci = CoherenceModel(
        model=model, 
        texts=texts, 
        dictionary=dictionary, 
        coherence='c_uci'
    )
    coherence_measures['C_uci'] = coherence_uci.get_coherence()
    
    # C_npmi coherence (normalized PMI)
    coherence_npmi = CoherenceModel(
        model=model, 
        texts=texts, 
        dictionary=dictionary, 
        coherence='c_npmi'
    )
    coherence_measures['C_npmi'] = coherence_npmi.get_coherence()
    
    return coherence_measures

def find_optimal_topics(processed_docs, topic_range=(2, 8)):
    """
    Find optimal number of topics by comparing coherence scores
    """
    results = []
    
    print("Finding optimal number of topics...")
    print("-" * 50)
    
    for num_topics in range(topic_range[0], topic_range[1] + 1):
        print(f"Evaluating {num_topics} topics...")
        
        # Create model
        model, dictionary, corpus = create_lda_model(processed_docs, num_topics)
        
        # Evaluate coherence
        coherence_scores = evaluate_coherence(model, processed_docs, dictionary, corpus)
        
        result = {
            'num_topics': num_topics,
            **coherence_scores
        }
        results.append(result)
        
        print(f"  C_v: {coherence_scores['C_v']:.4f}")
    
    return pd.DataFrame(results)

def display_topics_with_coherence(model, num_words=5):
    """
    Display topics with their top words
    """
    print("\nExtracted Topics:")
    print("=" * 60)
    
    for idx, topic in model.print_topics(num_words=num_words):
        print(f"Topic {idx + 1}:")
        # Clean up the topic string for better readability
        words = []
        for item in topic.split(' + '):
            word = item.split('*')[1].strip().replace('"', '')
            weight = float(item.split('*')[0])
            words.append(f"{word}({weight:.3f})")
        
        print(f"  {' | '.join(words)}")
        print()

def main_demo():
    """
    Main demonstration of topic coherence evaluation
    """
    print("TOPIC COHERENCE EVALUATION DEMO")
    print("=" * 60)
    print(f"Corpus: {len(sample_documents)} documents")
    
    # Preprocess documents
    processed_docs = preprocess_documents(sample_documents)
    print(f"Vocabulary size after preprocessing: {len(set(word for doc in processed_docs for word in doc))}")
    
    # Example 1: Single model evaluation
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Evaluating a 4-topic LDA model")
    print("=" * 60)
    
    model, dictionary, corpus = create_lda_model(processed_docs, num_topics=4)
    coherence_scores = evaluate_coherence(model, processed_docs, dictionary, corpus)
    
    display_topics_with_coherence(model)
    
    print("Coherence Scores:")
    print("-" * 30)
    for measure, score in coherence_scores.items():
        print(f"{measure:8}: {score:.4f}")
    
    # Example 2: Finding optimal number of topics
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Finding optimal number of topics")
    print("=" * 60)
    
    results_df = find_optimal_topics(processed_docs, topic_range=(2, 6))
    
    print("\nCoherence Comparison:")
    print(results_df.round(4))
    
    # Find best number of topics based on C_v score
    best_topics = results_df.loc[results_df['C_v'].idxmax(), 'num_topics']
    best_cv_score = results_df['C_v'].max()
    
    print(f"\nOptimal number of topics: {best_topics} (C_v score: {best_cv_score:.4f})")
    
    # Example 3: Coherence interpretation
    print("\n" + "=" * 60)
    print("COHERENCE MEASURES INTERPRETATION")
    print("=" * 60)
    
    interpretation = {
        'C_v': {
            'description': 'Most popular, correlates well with human judgment',
            'range': '[0, 1]',
            'interpretation': 'Higher is better. >0.5 is generally good'
        },
        'UMass': {
            'description': 'Intrinsic measure, based on document co-occurrence',
            'range': '(-∞, 14]',
            'interpretation': 'Higher is better. Values closer to 0 are better'
        },
        'C_uci': {
            'description': 'Based on PMI and sliding window',
            'range': '(-∞, ∞)',
            'interpretation': 'Higher is better'
        },
        'C_npmi': {
            'description': 'Normalized PMI measure',
            'range': '[-1, 1]',
            'interpretation': 'Higher is better. Closer to 1 is ideal'
        }
    }
    
    for measure, info in interpretation.items():
        print(f"\n{measure}:")
        print(f"  Description: {info['description']}")
        print(f"  Range: {info['range']}")
        print(f"  Interpretation: {info['interpretation']}")
    
    # Example 4: When to use coherence measures
    print("\n" + "=" * 60)
    print("WHEN TO USE TOPIC COHERENCE MEASURES")
    print("=" * 60)
    
    use_cases = [
        "1. Model Selection: Compare different topic modeling algorithms (LDA, NMF, etc.)",
        "2. Hyperparameter Tuning: Find optimal number of topics",
        "3. Preprocessing Impact: Evaluate effect of different text preprocessing steps",
        "4. Quality Assessment: Determine if topics are meaningful and interpretable",
        "5. Model Comparison: Compare models trained on different corpora",
        "6. Topic Filtering: Identify and remove low-quality topics"
    ]
    
    for use_case in use_cases:
        print(f"  {use_case}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("• Use C_v as primary measure (best human correlation)")
    print("• Use UMass for intrinsic evaluation (no external corpus needed)")
    print("• Compare multiple measures for robust evaluation")
    print("• Consider domain expertise alongside coherence scores")
    print("• Higher coherence doesn't always mean better topics for your use case")

if __name__ == "__main__":
    # Install required packages:
    # pip install gensim pandas matplotlib scikit-learn
    
    main_demo()
