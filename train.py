"""
Training Script for Word2Vec
============================
This script demonstrates how to train a Word2Vec model on text data.

Usage:
------
    # Train on sample text (quick test)
    python train.py --data sample --epochs 1000
    
    # Train on text8 dataset (full training)
    python train.py --data text8 --epochs 5
    
    # Train with custom parameters
    python train.py --data text8 --embedding-dim 200 --window-size 8 --negative-samples 10
"""

import numpy as np
import argparse
import os
import pickle
from typing import List

from word2vec import Word2Vec, Word2VecCBOW
from preprocessing import (
    preprocess_sentences,
    download_text8_dataset,
    load_sample_text,
    split_into_sentences
)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)


def find_similar_words(
    model: Word2Vec,
    vocab,
    word: str,
    top_k: int = 10
) -> List[tuple]:
    """
    Find the top-k most similar words to a given word.
    
    Parameters:
    -----------
    model : Word2Vec
        Trained model
    vocab : Vocabulary
        Vocabulary object
    word : str
        Query word
    top_k : int
        Number of similar words to return
    
    Returns:
    --------
    similar_words : list of (word, similarity) tuples
    """
    if word not in vocab.word2id:
        print(f"Word '{word}' not in vocabulary")
        return []
    
    word_id = vocab.word2id[word]
    word_vec = model.get_word_vector(word_id)
    
    similarities = []
    for other_id in range(len(vocab)):
        if other_id != word_id and other_id not in [vocab.word2id.get(vocab.PAD_TOKEN), 
                                                      vocab.word2id.get(vocab.UNK_TOKEN)]:
            other_vec = model.get_word_vector(other_id)
            sim = cosine_similarity(word_vec, other_vec)
            similarities.append((vocab.decode(other_id), sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


def analogies(
    model: Word2Vec,
    vocab,
    word_a: str,
    word_b: str,
    word_c: str,
    top_k: int = 5
) -> List[tuple]:
    """
    Solve analogy: word_a is to word_b as word_c is to ?
    
    Example: king - man + woman = queen
    
    Parameters:
    -----------
    model : Word2Vec
        Trained model
    vocab : Vocabulary
        Vocabulary object
    word_a, word_b, word_c : str
        Words in the analogy
    top_k : int
        Number of results to return
    
    Returns:
    --------
    results : list of (word, similarity) tuples
    """
    for w in [word_a, word_b, word_c]:
        if w not in vocab.word2id:
            print(f"Word '{w}' not in vocabulary")
            return []
    
    vec_a = model.get_word_vector(vocab.word2id[word_a])
    vec_b = model.get_word_vector(vocab.word2id[word_b])
    vec_c = model.get_word_vector(vocab.word2id[word_c])
    
    # target = vec_b - vec_a + vec_c
    target = vec_b - vec_a + vec_c
    
    similarities = []
    for other_id in range(len(vocab)):
        other_word = vocab.decode(other_id)
        if other_word not in [word_a, word_b, word_c, vocab.PAD_TOKEN, vocab.UNK_TOKEN]:
            other_vec = model.get_word_vector(other_id)
            sim = cosine_similarity(target, other_vec)
            similarities.append((other_word, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def train_word2vec(
    data_source: str = 'sample',
    model_type: str = 'skipgram',
    embedding_dim: int = 100,
    window_size: int = 5,
    negative_samples: int = 5,
    learning_rate: float = 0.025,
    epochs: int = 5,
    min_count: int = 5,
    subsampling_threshold: float = 1e-5,
    output_dir: str = 'output',
    seed: int = 42,
    max_pairs_per_epoch: int = None
):
    """
    Train a Word2Vec model.
    
    Parameters:
    -----------
    data_source : str
        'sample' or 'text8'
    model_type : str
        'skipgram' or 'cbow'
    embedding_dim : int
        Dimension of word embeddings
    window_size : int
        Context window size
    negative_samples : int
        Number of negative samples
    learning_rate : float
        Initial learning rate
    epochs : int
        Number of training epochs
    min_count : int
        Minimum word frequency
    subsampling_threshold : float
        Threshold for subsampling
    output_dir : str
        Directory to save outputs
    seed : int
        Random seed
    max_pairs_per_epoch : int
        Maximum training pairs per epoch (None = auto for text8)
    """
    print("=" * 60)
    print("Word2Vec Training")
    print("=" * 60)
    
    # Load data
    print(f"\n[1/5] Loading data: {data_source}")
    if data_source == 'text8':
        text = download_text8_dataset('data/text8.txt')
        # Split text8 into chunks (it's one long string)
        words = text.split()
        chunk_size = 1000  # words per "sentence"
        sentences_text = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    else:
        text = load_sample_text()
        sentences_text = split_into_sentences(text)
    
    print(f"Number of sentences: {len(sentences_text)}")
    
    # Preprocess
    print(f"\n[2/5] Preprocessing (min_count={min_count})...")
    vocab, sentences = preprocess_sentences(sentences_text, min_count=min_count)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Total sentences after filtering: {len(sentences)}")
    
    # Build negative sampling table
    print(f"\n[3/5] Building negative sampling table...")
    word_counts = vocab.get_word_counts()
    
    # Initialize model
    print(f"\n[4/5] Initializing {model_type} model...")
    model_class = Word2VecCBOW if model_type == 'cbow' else Word2Vec
    model = model_class(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        window_size=window_size,
        negative_samples=negative_samples,
        learning_rate=learning_rate,
        subsampling_threshold=subsampling_threshold,
        seed=seed
    )
    model.build_negative_sampling_table(word_counts)
    
    print(f"Model parameters:")
    print(f"  - Vocabulary size: {len(vocab)}")
    print(f"  - Embedding dimension: {embedding_dim}")
    print(f"  - Window size: {window_size}")
    print(f"  - Negative samples: {negative_samples}")
    print(f"  - Initial learning rate: {learning_rate}")
    print(f"  - Model type: {model_type}")
    
    # Train
    print(f"\n[5/5] Training for {epochs} epochs...")
    print("-" * 60)
    
    max_pairs = max_pairs_per_epoch
    if max_pairs:
        print(f"Note: Limiting to {max_pairs:,} training pairs per epoch")

    batch_size = 256 if data_source == 'text8' else 64
    model.train(sentences, epochs=epochs, verbose=True, max_pairs_per_epoch=max_pairs, batch_size=batch_size)
    print("-" * 60)
    
    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, f'word2vec_{model_type}.pkl')
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save vocabulary
    vocab_path = os.path.join(output_dir, 'vocab.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary saved to: {vocab_path}")
    
    # Save embeddings as text (for inspection)
    embeddings_path = os.path.join(output_dir, 'embeddings.txt')
    with open(embeddings_path, 'w', encoding='utf-8') as f:
        f.write(f"{len(vocab)} {embedding_dim}\n")
        for word_id in range(len(vocab)):
            word = vocab.decode(word_id)
            vec = model.get_word_vector(word_id)
            vec_str = ' '.join(map(str, vec))
            f.write(f"{word} {vec_str}\n")
    print(f"Embeddings saved to: {embeddings_path}")
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)
    
    # Find similar words
    test_words = ['learning', 'intelligence', 'data', 'neural']
    if data_source == 'sample':
        test_words = ['learning', 'intelligence', 'data']
    
    print("\nMost similar words:")
    for word in test_words:
        if word in vocab.word2id:
            similar = find_similar_words(model, vocab, word, top_k=5)
            print(f"\n  {word}:")
            for sim_word, sim in similar:
                print(f"    - {sim_word}: {sim:.4f}")
    
    # Analogies
    print("\nAnalogies:")
    analogy_tests = [
        ('learning', 'machine', 'intelligence'),  # machine : learning :: ? : intelligence
    ]
    
    for a, b, c in analogy_tests:
        if all(w in vocab.word2id for w in [a, b, c]):
            results = analogies(model, vocab, a, b, c, top_k=5)
            print(f"\n  {b} - {a} + {c} = ?")
            for word, sim in results:
                print(f"    - {word}: {sim:.4f}")
    
    return model, vocab


def main():
    parser = argparse.ArgumentParser(description='Train Word2Vec model')
    parser.add_argument('--data', type=str, default='sample',
                        choices=['sample', 'text8'],
                        help='Dataset to use')
    parser.add_argument('--model-type', type=str, default='skipgram',
                        choices=['skipgram', 'cbow'],
                        help='Model architecture')
    parser.add_argument('--embedding-dim', type=int, default=100,
                        help='Embedding dimension')
    parser.add_argument('--window-size', type=int, default=5,
                        help='Context window size')
    parser.add_argument('--negative-samples', type=int, default=5,
                        help='Number of negative samples')
    parser.add_argument('--learning-rate', type=float, default=0.025,
                        help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--min-count', type=int, default=5,
                        help='Minimum word frequency')
    parser.add_argument('--subsampling-threshold', type=float, default=1e-5,
                        help='Subsampling threshold')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--max-pairs', type=int, default=None,
                        help='Max training pairs per epoch (default: 1M for text8, unlimited for sample)')
    
    args = parser.parse_args()
    
    train_word2vec(
        data_source=args.data,
        model_type=args.model_type,
        embedding_dim=args.embedding_dim,
        window_size=args.window_size,
        negative_samples=args.negative_samples,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        min_count=args.min_count,
        subsampling_threshold=args.subsampling_threshold,
        output_dir=args.output_dir,
        seed=args.seed,
        max_pairs_per_epoch=args.max_pairs
    )


if __name__ == '__main__':
    main()
