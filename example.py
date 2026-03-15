"""
Simple Example: Word2Vec in Pure NumPy
======================================

This is a minimal example demonstrating the core Word2Vec implementation.
Run this to verify the implementation works correctly.
"""

import numpy as np
from word2vec import Word2Vec
from preprocessing import preprocess_sentences, split_into_sentences

# Sample corpus (small for demonstration)
corpus = """
Machine learning is amazing. Deep learning is a subset of machine learning.
Neural networks power deep learning models. Natural language processing uses machine learning.
Word embeddings capture semantic meaning. Word2Vec learns word embeddings efficiently.
Data science combines statistics and machine learning. Artificial intelligence includes machine learning.
Neural networks are inspired by the brain. Deep neural networks have many layers.
"""

print("=" * 60)
print("Word2Vec Minimal Example")
print("=" * 60)

# Step 1: Preprocess
print("\n[Step 1] Preprocessing text...")
sentences_text = split_into_sentences(corpus)
vocab, sentences = preprocess_sentences(sentences_text, min_count=1)
print(f"Vocabulary size: {len(vocab)}")
print(f"Sentences: {len(sentences)}")

# Step 2: Initialize model
print("\n[Step 2] Initializing Word2Vec model...")
model = Word2Vec(
    vocab_size=len(vocab),
    embedding_dim=10,  # Small for quick demo
    window_size=2,
    negative_samples=3,
    learning_rate=0.025,
    subsampling_threshold=0  # Disable subsampling for small corpus
)

# Step 3: Build negative sampling table
print("\n[Step 3] Building negative sampling table...")
word_counts = vocab.get_word_counts()
model.build_negative_sampling_table(word_counts, table_size=1000)

# Step 4: Train
print("\n[Step 4] Training for 100 epochs...")
model.train(sentences, epochs=100, verbose=False)
print(f"Final loss: {model.loss_history[-1]:.4f}")

# Step 5: Evaluate
print("\n[Step 5] Evaluation - Finding similar words...")

def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def find_similar(word, top_k=3):
    if word not in vocab.word2id:
        return []
    word_id = vocab.word2id[word]
    word_vec = model.get_word_vector(word_id)
    
    similarities = []
    for other_id in range(len(vocab)):
        if other_id != word_id:
            other_vec = model.get_word_vector(other_id)
            sim = cosine_similarity(word_vec, other_vec)
            similarities.append((vocab.decode(other_id), sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# Test similar words
test_words = ['learning', 'machine', 'neural']
for word in test_words:
    similar = find_similar(word)
    print(f"\n  Words most similar to '{word}':")
    for sim_word, sim in similar:
        print(f"    - {sim_word}: {sim:.4f}")

print("\n" + "=" * 60)
print("Example completed successfully!")
print("=" * 60)
