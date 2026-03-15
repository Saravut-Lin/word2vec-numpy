"""
Data Preprocessing for Word2Vec
===============================
This module handles text preprocessing, vocabulary building, and dataset preparation.
"""

import numpy as np
import re
from collections import Counter
from typing import List, Tuple, Dict, Set
import urllib.request
import os


class Vocabulary:
    """
    Vocabulary manager for Word2Vec.
    
    Handles word-to-id mapping, frequency counting, and filtering.
    """
    
    def __init__(self, min_count: int = 5):
        """
        Parameters:
        -----------
        min_count : int
            Minimum word frequency to include in vocabulary
        """
        self.min_count = min_count
        self.word2id: Dict[str, int] = {}
        self.id2word: Dict[int, str] = {}
        self.word_counts: Counter = Counter()
        self.word_counts_filtered: Dict[int, int] = {}
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        
    def build_vocab(self, sentences: List[List[str]]) -> 'Vocabulary':
        """
        Build vocabulary from tokenized sentences.
        
        Parameters:
        -----------
        sentences : list of list of str
            Tokenized sentences
        
        Returns:
        --------
        self : Vocabulary
        """
        # Count word frequencies
        for sentence in sentences:
            self.word_counts.update(sentence)
        
        # Filter by minimum count and create mappings
        filtered_words = [
            word for word, count in self.word_counts.items()
            if count >= self.min_count
        ]
        
        # Add special tokens first
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN]
        all_words = special_tokens + filtered_words
        
        self.word2id = {word: i for i, word in enumerate(all_words)}
        self.id2word = {i: word for word, i in self.word2id.items()}
        
        # Create filtered word counts with IDs
        self.word_counts_filtered = {
            self.word2id[word]: self.word_counts[word]
            for word in filtered_words
            if word in self.word2id
        }
        
        print(f"Vocabulary size: {len(self.word2id)}")
        print(f"Total words in corpus: {sum(self.word_counts.values())}")
        print(f"Words filtered (min_count={self.min_count}): {len(filtered_words)}")
        
        return self
    
    def encode(self, word: str) -> int:
        """Convert word to ID."""
        return self.word2id.get(word, self.word2id[self.UNK_TOKEN])
    
    def decode(self, word_id: int) -> str:
        """Convert ID to word."""
        return self.id2word.get(word_id, self.UNK_TOKEN)
    
    def encode_sentence(self, sentence: List[str]) -> List[int]:
        """Convert sentence (list of words) to list of IDs."""
        return [self.encode(word) for word in sentence]
    
    def decode_sentence(self, word_ids: List[int]) -> List[str]:
        """Convert list of IDs to sentence (list of words)."""
        return [self.decode(word_id) for word_id in word_ids]
    
    def __len__(self) -> int:
        return len(self.word2id)
    
    def get_word_counts(self) -> Dict[int, int]:
        """Get word counts as dictionary of word_id -> count."""
        return self.word_counts_filtered


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    """
    Simple tokenization: lowercase, remove non-alphanumeric, split on whitespace.
    
    Parameters:
    -----------
    text : str
        Input text
    lowercase : bool
        Whether to convert to lowercase
    
    Returns:
    --------
    tokens : list of str
    """
    if lowercase:
        text = text.lower()
    
    # Replace non-alphanumeric with spaces (keep apostrophes for contractions)
    text = re.sub(r"[^a-zA-Z0-9'\s]", ' ', text)
    
    # Split on whitespace
    tokens = text.split()
    
    # Remove empty tokens and standalone apostrophes
    tokens = [t.strip("'") for t in tokens if t.strip("'")]
    
    return tokens


def preprocess_sentences(
    texts: List[str],
    min_count: int = 5,
    lowercase: bool = True
) -> Tuple[Vocabulary, List[List[int]]]:
    """
    Preprocess a list of texts into tokenized, numericalized sentences.
    
    Parameters:
    -----------
    texts : list of str
        Raw text documents
    min_count : int
        Minimum word frequency for vocabulary
    lowercase : bool
        Whether to lowercase text
    
    Returns:
    --------
    vocab : Vocabulary
        Built vocabulary object
    sentences : list of list of int
        Numericalized sentences
    """
    # Tokenize all texts
    tokenized_sentences = [tokenize(text, lowercase) for text in texts]
    
    # Filter out empty sentences
    tokenized_sentences = [s for s in tokenized_sentences if len(s) > 0]
    
    # Build vocabulary
    vocab = Vocabulary(min_count=min_count)
    vocab.build_vocab(tokenized_sentences)
    
    # Convert to IDs
    sentences = [vocab.encode_sentence(sent) for sent in tokenized_sentences]
    
    # Filter out sentences with only UNK tokens
    unk_id = vocab.word2id[vocab.UNK_TOKEN]
    sentences = [s for s in sentences if any(w != unk_id for w in s)]
    
    return vocab, sentences


def download_text8_dataset(save_path: str = 'data/text8.txt') -> str:
    """
    Download the text8 dataset (first 100MB of cleaned Wikipedia text).
    
    This is a standard benchmark dataset for Word2Vec.
    
    Parameters:
    -----------
    save_path : str
        Path to save the dataset
    
    Returns:
    --------
    text : str
        The downloaded text
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if os.path.exists(save_path):
        print(f"Loading existing dataset from {save_path}")
        with open(save_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    url = 'http://mattmahoney.net/dc/text8.zip'
    zip_path = save_path + '.zip'
    
    print(f"Downloading text8 dataset from {url}...")
    urllib.request.urlretrieve(url, zip_path)
    
    print("Extracting...")
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(save_path))
    
    os.remove(zip_path)
    
    # The extracted file is named 'text8'
    extracted_path = os.path.join(os.path.dirname(save_path), 'text8')
    os.rename(extracted_path, save_path)
    
    with open(save_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Dataset loaded: {len(text)} characters, {len(text.split())} words")
    return text


def load_sample_text() -> str:
    """
    Return a small sample text for quick testing.
    """
    return """
    Machine learning is a subset of artificial intelligence that enables computers 
    to learn and improve from experience without being explicitly programmed. 
    Deep learning is a subset of machine learning that uses neural networks with 
    many layers to model and understand complex patterns in data.
    
    Natural language processing is a branch of artificial intelligence that helps 
    computers understand, interpret and manipulate human language. NLP draws from 
    many disciplines, including computer science and computational linguistics, 
    in its pursuit to fill the gap between human communication and computer understanding.
    
    Word embeddings are a type of word representation that allows words with similar 
    meaning to have a similar representation. Word2Vec is a popular algorithm for 
    learning word embeddings from raw text. It was developed by researchers at Google.
    
    Neural networks are computing systems inspired by the biological neural networks 
    that constitute animal brains. Such systems learn to perform tasks by considering 
    examples, generally without being programmed with task-specific rules.
    
    Data science is an inter-disciplinary field that uses scientific methods, processes, 
    algorithms and systems to extract knowledge and insights from structured and 
    unstructured data. Data science is related to data mining and big data.
    
    Artificial intelligence is intelligence demonstrated by machines, as opposed to 
    the natural intelligence displayed by humans and animals. AI research has been 
    defined as the field of study of intelligent agents, which refers to any system 
    that perceives its environment and takes actions that maximize its chance of 
    achieving its goals.
    """


def create_sliding_windows(
    sentences: List[List[int]],
    window_size: int = 5
) -> List[Tuple[int, int]]:
    """
    Create (center, context) pairs using sliding windows.
    
    Parameters:
    -----------
    sentences : list of list of int
        Numericalized sentences
    window_size : int
        Size of context window on each side
    
    Returns:
    --------
    pairs : list of (center, context) tuples
    """
    pairs = []
    for sentence in sentences:
        for i in range(len(sentence)):
            center = sentence[i]
            # Left context
            for j in range(max(0, i - window_size), i):
                pairs.append((center, sentence[j]))
            # Right context
            for j in range(i + 1, min(len(sentence), i + window_size + 1)):
                pairs.append((center, sentence[j]))
    return pairs


def split_into_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter.
    """
    # Split on sentence-ending punctuation
    sentences = re.split(r'[.!?]+', text)
    # Clean and filter
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


if __name__ == '__main__':
    # Test the preprocessing
    sample_text = load_sample_text()
    sentences_text = split_into_sentences(sample_text)
    
    vocab, sentences = preprocess_sentences(sentences_text, min_count=1)
    
    print(f"\nSample sentences (first 3):")
    for i, sent in enumerate(sentences[:3]):
        print(f"  {i+1}. {vocab.decode_sentence(sent)}")
    
    print(f"\nVocabulary sample:")
    for i, (word, word_id) in enumerate(list(vocab.word2id.items())[:10]):
        print(f"  {word}: {word_id}")
