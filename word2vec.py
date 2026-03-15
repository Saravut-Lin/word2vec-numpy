"""
Word2Vec Implementation in Pure NumPy
======================================
This module implements the Skip-gram Word2Vec model with Negative Sampling.

The skip-gram model predicts context words given a center word.
Negative sampling approximates the softmax by sampling negative examples.

Mathematical Foundation:
------------------------
For a center word w_c and context word w_o, the objective is to maximize:
    log(sigmoid(u_o^T v_c)) + sum_{i=1}^k log(sigmoid(-u_{n_i}^T v_c))

where:
- v_c: input vector for center word (shape: embedding_dim)
- u_o: output vector for positive context word (shape: embedding_dim)
- u_{n_i}: output vectors for k negative samples
- sigmoid(x) = 1 / (1 + exp(-x))

Gradients:
----------
For positive sample (context word):
    dL/dv_c = (sigmoid(u_o^T v_c) - 1) * u_o
    dL/du_o = (sigmoid(u_o^T v_c) - 1) * v_c

For negative samples:
    dL/dv_c = sigmoid(u_{n_i}^T v_c) * u_{n_i}
    dL/du_{n_i} = sigmoid(u_{n_i}^T v_c) * v_c
"""

import numpy as np
from collections import Counter
from typing import List, Tuple, Dict, Optional
import pickle


class Word2Vec:
    """
    Skip-gram Word2Vec model with Negative Sampling.
    
    Parameters:
    -----------
    vocab_size : int
        Size of the vocabulary
    embedding_dim : int
        Dimension of word embeddings
    window_size : int
        Context window size (words on each side of center word)
    negative_samples : int
        Number of negative samples per positive pair
    learning_rate : float
        Learning rate for SGD updates
    subsampling_threshold : float
        Threshold for subsampling frequent words (recommended: 1e-5)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        window_size: int = 5,
        negative_samples: int = 5,
        learning_rate: float = 0.025,
        subsampling_threshold: float = 1e-5,
        seed: int = 42
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.negative_samples = negative_samples
        self.learning_rate = learning_rate
        self.subsampling_threshold = subsampling_threshold
        self.seed = seed
        
        # Initialize random seed for reproducibility
        np.random.seed(seed)
        
        # Initialize weight matrices
        # W_in: input embeddings (center words) - shape: (vocab_size, embedding_dim)
        # Initialized with uniform distribution in [-0.5/emb_dim, 0.5/emb_dim]
        self.W_in = np.random.uniform(
            -0.5 / embedding_dim,
            0.5 / embedding_dim,
            (vocab_size, embedding_dim)
        ).astype(np.float32)
        
        # W_out: output embeddings (context words) - shape: (vocab_size, embedding_dim)
        # Initialized with zeros (common practice for output layer)
        self.W_out = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
        
        # For negative sampling: unigram distribution raised to 3/4 power
        self.neg_sampling_table = None
        
        # Word frequency counts (needed for subsampling and negative sampling)
        self.word_counts = None
        self.total_words = 0
        
        # Training history
        self.loss_history = []
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Numerically stable sigmoid function.
        
        For large positive x: sigmoid(x) ≈ 1
        For large negative x: sigmoid(x) ≈ 0
        We clip x to [-6, 6] to avoid overflow.
        """
        x = np.clip(x, -6, 6)
        return 1 / (1 + np.exp(-x))
    
    def build_negative_sampling_table(self, word_counts: Dict[int, int], table_size: int = 100000000):
        """
        Build a table for negative sampling using the unigram distribution
        raised to the 3/4 power.
        
        The 3/4 power smooths the distribution, giving less frequent words
        a slightly higher probability of being sampled.
        
        Parameters:
        -----------
        word_counts : dict mapping word_id -> count
        table_size : int
            Size of the sampling table (larger = more accurate sampling)
        """
        self.word_counts = word_counts
        self.total_words = sum(word_counts.values())
        
        # Calculate sampling probabilities: P(w) = count(w)^0.75 / sum(count(w')^0.75)
        words = list(word_counts.keys())
        counts = np.array([word_counts[w] for w in words], dtype=np.float64)
        
        # Raise to 3/4 power and normalize
        probabilities = np.power(counts, 0.75)
        probabilities /= probabilities.sum()
        
        # Create sampling table
        # Each word appears proportional to its probability
        self.neg_sampling_table = np.zeros(table_size, dtype=np.int32)
        cumsum = np.cumsum(probabilities)
        
        idx = 0
        for i, word_id in enumerate(words):
            # Fill table slots for this word
            num_slots = int(probabilities[i] * table_size)
            if idx + num_slots <= table_size:
                self.neg_sampling_table[idx:idx + num_slots] = word_id
                idx += num_slots
        
        # Shuffle to randomize
        np.random.shuffle(self.neg_sampling_table)
    
    def get_negative_samples(self, batch_size: int, exclude_words: np.ndarray) -> np.ndarray:
        """
        Sample negative examples from the negative sampling table.
        
        Parameters:
        -----------
        batch_size : int
            Number of negative samples needed per positive example
        exclude_words : np.ndarray
            Word IDs to exclude (positive samples)
        
        Returns:
        --------
        negative_samples : np.ndarray of shape (len(exclude_words), negative_samples)
        """
        n = len(exclude_words)
        negatives = np.zeros((n, batch_size), dtype=np.int32)
        
        for i in range(n):
            # Sample negative words
            samples = self.neg_sampling_table[
                np.random.randint(0, len(self.neg_sampling_table), batch_size)
            ]
            
            # Resample if we accidentally sampled the positive word
            while np.any(samples == exclude_words[i]):
                mask = samples == exclude_words[i]
                samples[mask] = self.neg_sampling_table[
                    np.random.randint(0, len(self.neg_sampling_table), mask.sum())
                ]
            
            negatives[i] = samples
        
        return negatives
    
    def subsample_words(self, word_ids: List[int]) -> List[int]:
        """
        Subsample frequent words according to the paper's formula:
        P(discard w) = 1 - sqrt(t / f(w))
        
        where t is the threshold and f(w) is the word frequency.
        
        Parameters:
        -----------
        word_ids : list of word IDs
        
        Returns:
        --------
        subsampled : list of word IDs after subsampling
        """
        if self.word_counts is None or self.subsampling_threshold <= 0:
            return word_ids
        
        subsampled = []
        for word_id in word_ids:
            word_count = self.word_counts.get(word_id, 0)
            # Skip words not in word_counts (e.g., UNK tokens) - always keep them
            if word_count == 0:
                subsampled.append(word_id)
                continue
            
            freq = word_count / self.total_words
            # Probability of keeping the word
            keep_prob = np.sqrt(self.subsampling_threshold / freq)
            if np.random.random() < keep_prob:
                subsampled.append(word_id)
        
        return subsampled
    
    def generate_training_pairs(
        self,
        word_ids: List[int]
    ) -> List[Tuple[int, int]]:
        """
        Generate (center_word, context_word) pairs from a sentence.
        
        Parameters:
        -----------
        word_ids : list of word IDs
        
        Returns:
        --------
        pairs : list of (center_word_id, context_word_id) tuples
        """
        pairs = []
        sentence_length = len(word_ids)
        
        for center_pos in range(sentence_length):
            center_word = word_ids[center_pos]
            
            # Define context window (random size for better training)
            window = np.random.randint(1, self.window_size + 1)
            
            # Get context words within the window
            start = max(0, center_pos - window)
            end = min(sentence_length, center_pos + window + 1)
            
            for context_pos in range(start, end):
                if context_pos != center_pos:
                    context_word = word_ids[context_pos]
                    pairs.append((center_word, context_word))
        
        return pairs
    
    def forward(self, center_word: int, context_word: int, negative_words: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass for a single training example.
        
        Parameters:
        -----------
        center_word : int
            ID of the center word
        context_word : int
            ID of the positive context word
        negative_words : np.ndarray
            IDs of negative sampled words (shape: negative_samples,)
        
        Returns:
        --------
        center_vec : np.ndarray
            Input vector for center word (shape: embedding_dim,)
        context_out : np.ndarray
            Output vector for context word (shape: embedding_dim,)
        negative_out : np.ndarray
            Output vectors for negative words (shape: negative_samples, embedding_dim)
        """
        # Get input vector for center word
        center_vec = self.W_in[center_word]  # Shape: (embedding_dim,)
        
        # Get output vector for positive context word
        context_out = self.W_out[context_word]  # Shape: (embedding_dim,)
        
        # Get output vectors for negative samples
        negative_out = self.W_out[negative_words]  # Shape: (negative_samples, embedding_dim)
        
        return center_vec, context_out, negative_out
    
    def compute_loss(
        self,
        center_vec: np.ndarray,
        context_out: np.ndarray,
        negative_out: np.ndarray
    ) -> float:
        """
        Compute the negative sampling loss for a single training example.
        
        Loss = -log(sigmoid(u_o^T v_c)) - sum_{i=1}^k log(sigmoid(-u_{n_i}^T v_c))
        
        Parameters:
        -----------
        center_vec : np.ndarray
            Input vector for center word
        context_out : np.ndarray
            Output vector for positive context word
        negative_out : np.ndarray
            Output vectors for negative samples
        
        Returns:
        --------
        loss : float
            The computed loss value
        """
        # Positive sample score: u_o^T v_c
        pos_score = np.dot(context_out, center_vec)
        pos_loss = -np.log(self._sigmoid(pos_score) + 1e-10)
        
        # Negative samples score: u_{n_i}^T v_c
        neg_scores = np.dot(negative_out, center_vec)
        neg_loss = -np.sum(np.log(self._sigmoid(-neg_scores) + 1e-10))
        
        return pos_loss + neg_loss
    
    def backward(
        self,
        center_word: int,
        context_word: int,
        negative_words: np.ndarray,
        center_vec: np.ndarray,
        context_out: np.ndarray,
        negative_out: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass: compute gradients for a single training example.
        
        Gradient Derivation:
        --------------------
        Let z_pos = u_o^T v_c (positive score)
        Let z_neg_i = u_{n_i}^T v_c (negative scores)
        
        For positive sample:
            dL/dz_pos = sigmoid(z_pos) - 1
            dL/dv_c = (sigmoid(z_pos) - 1) * u_o
            dL/du_o = (sigmoid(z_pos) - 1) * v_c
        
        For negative samples:
            dL/dz_neg_i = sigmoid(z_neg_i)
            dL/dv_c += sigmoid(z_neg_i) * u_{n_i}
            dL/du_{n_i} = sigmoid(z_neg_i) * v_c
        
        Parameters:
        -----------
        center_word : int
            ID of center word
        context_word : int
            ID of positive context word
        negative_words : np.ndarray
            IDs of negative sampled words
        center_vec : np.ndarray
            Input vector for center word
        context_out : np.ndarray
            Output vector for positive context word
        negative_out : np.ndarray
            Output vectors for negative samples
        
        Returns:
        --------
        grad_center : np.ndarray
            Gradient for center word input vector
        grad_context : np.ndarray
            Gradient for context word output vector
        grad_negative : np.ndarray
            Gradients for negative word output vectors
        """
        # Positive sample gradients
        pos_score = np.dot(context_out, center_vec)
        pos_sigmoid = self._sigmoid(pos_score)
        
        # dL/dz_pos = sigmoid(z_pos) - 1
        pos_grad_coef = pos_sigmoid - 1  # Scalar
        
        # Gradient for center vector from positive sample
        grad_center = pos_grad_coef * context_out  # Shape: (embedding_dim,)
        
        # Gradient for context output vector
        grad_context = pos_grad_coef * center_vec  # Shape: (embedding_dim,)
        
        # Negative samples gradients
        neg_scores = np.dot(negative_out, center_vec)  # Shape: (negative_samples,)
        neg_sigmoids = self._sigmoid(neg_scores)  # Shape: (negative_samples,)
        
        # dL/dz_neg_i = sigmoid(z_neg_i)
        neg_grad_coefs = neg_sigmoids  # Shape: (negative_samples,)
        
        # Gradient for center vector from negative samples
        grad_center += np.dot(neg_grad_coefs, negative_out)  # Shape: (embedding_dim,)
        
        # Gradients for negative output vectors
        # Each row is grad for one negative sample: sigmoid(z_neg_i) * v_c
        grad_negative = np.outer(neg_grad_coefs, center_vec)  # Shape: (negative_samples, embedding_dim)
        
        return grad_center, grad_context, grad_negative
    
    def update_parameters(
        self,
        center_word: int,
        context_word: int,
        negative_words: np.ndarray,
        grad_center: np.ndarray,
        grad_context: np.ndarray,
        grad_negative: np.ndarray
    ):
        """
        Update parameters using SGD.
        
        Parameters:
        -----------
        center_word : int
            ID of center word
        context_word : int
            ID of positive context word
        negative_words : np.ndarray
            IDs of negative sampled words
        grad_center : np.ndarray
            Gradient for center word input vector
        grad_context : np.ndarray
            Gradient for context word output vector
        grad_negative : np.ndarray
            Gradients for negative word output vectors
        """
        # Update center word input vector
        self.W_in[center_word] -= self.learning_rate * grad_center
        
        # Update context word output vector
        self.W_out[context_word] -= self.learning_rate * grad_context
        
        # Update negative word output vectors
        for i, neg_word in enumerate(negative_words):
            self.W_out[neg_word] -= self.learning_rate * grad_negative[i]
    
    def train_step(
        self,
        center_word: int,
        context_word: int
    ) -> float:
        """
        Perform one training step (forward + backward + update).
        
        Parameters:
        -----------
        center_word : int
            ID of center word
        context_word : int
            ID of context word
        
        Returns:
        --------
        loss : float
            Loss for this training example
        """
        # Get negative samples
        negative_words = self.get_negative_samples(
            self.negative_samples,
            np.array([center_word])
        )[0]
        
        # Forward pass
        center_vec, context_out, negative_out = self.forward(
            center_word, context_word, negative_words
        )
        
        # Compute loss
        loss = self.compute_loss(center_vec, context_out, negative_out)
        
        # Backward pass
        grad_center, grad_context, grad_negative = self.backward(
            center_word, context_word, negative_words,
            center_vec, context_out, negative_out
        )
        
        # Update parameters
        self.update_parameters(
            center_word, context_word, negative_words,
            grad_center, grad_context, grad_negative
        )
        
        return loss
    
    def train_batch(
        self,
        center_words: np.ndarray,
        context_words: np.ndarray
    ) -> float:
        """
        Perform a batched training step on multiple (center, context) pairs.

        Parameters:
        -----------
        center_words : np.ndarray of shape (batch_size,)
        context_words : np.ndarray of shape (batch_size,)

        Returns:
        --------
        total_loss : float
        """
        batch_size = len(center_words)
        k = self.negative_samples

        # Get negative samples for each pair: (batch_size, k)
        neg_words = self.get_negative_samples(k, center_words)

        # Lookup vectors
        v_c = self.W_in[center_words]       # (batch_size, dim)
        u_o = self.W_out[context_words]      # (batch_size, dim)
        u_neg = self.W_out[neg_words]        # (batch_size, k, dim)

        # Positive scores
        pos_scores = np.sum(v_c * u_o, axis=1)          # (batch_size,)
        pos_sig = self._sigmoid(pos_scores)              # (batch_size,)
        pos_loss = -np.log(pos_sig + 1e-10)

        # Negative scores
        neg_scores = np.einsum('bd,bkd->bk', v_c, u_neg) # (batch_size, k)
        neg_sig = self._sigmoid(neg_scores)               # (batch_size, k)
        neg_loss = -np.sum(np.log(self._sigmoid(-neg_scores) + 1e-10), axis=1)

        total_loss = np.sum(pos_loss + neg_loss)

        # Gradients
        pos_coef = (pos_sig - 1)[:, np.newaxis]           # (batch_size, 1)
        grad_v_c = pos_coef * u_o                          # (batch_size, dim)
        grad_u_o = pos_coef * v_c                          # (batch_size, dim)

        neg_coef = neg_sig[:, :, np.newaxis]               # (batch_size, k, 1)
        grad_v_c += np.sum(neg_coef * u_neg, axis=1)       # (batch_size, dim)
        grad_u_neg = neg_coef * v_c[:, np.newaxis, :]      # (batch_size, k, dim)

        # Update W_in for center words
        lr = self.learning_rate
        np.subtract.at(self.W_in, center_words, lr * grad_v_c)

        # Update W_out for context words
        np.subtract.at(self.W_out, context_words, lr * grad_u_o)

        # Update W_out for negative words
        neg_flat = neg_words.ravel()
        grad_neg_flat = (lr * grad_u_neg).reshape(-1, self.embedding_dim)
        np.subtract.at(self.W_out, neg_flat, grad_neg_flat)

        return total_loss

    def train(
        self,
        sentences: List[List[int]],
        epochs: int = 5,
        verbose: bool = True,
        max_pairs_per_epoch: int = None,
        report_interval: int = 100000,
        batch_size: int = 64
    ):
        """
        Train the Word2Vec model on a corpus.

        Parameters:
        -----------
        sentences : list of list of int
            Tokenized sentences (each sentence is a list of word IDs)
        epochs : int
            Number of training epochs
        verbose : bool
            Whether to print progress
        max_pairs_per_epoch : int
            Maximum training pairs to process per epoch (None = no limit)
        report_interval : int
            Print progress every N training steps
        batch_size : int
            Number of pairs to process in each batch
        """
        # Pre-calculate total word count for learning rate decay
        total_words = sum(len(s) for s in sentences)
        estimated_total_pairs = total_words * self.window_size * 2
        total_pairs_processed = 0
        initial_lr = self.learning_rate

        for epoch in range(epochs):
            epoch_loss = 0.0
            pair_count = 0
            last_report = 0

            # Collect pairs in a buffer and process in batches
            center_buf = []
            context_buf = []

            for sentence in sentences:
                subsampled = self.subsample_words(sentence)
                pairs = self.generate_training_pairs(subsampled)

                for center_word, context_word in pairs:
                    center_buf.append(center_word)
                    context_buf.append(context_word)

                    if len(center_buf) >= batch_size:
                        loss = self.train_batch(
                            np.array(center_buf, dtype=np.int32),
                            np.array(context_buf, dtype=np.int32)
                        )
                        epoch_loss += loss
                        pair_count += len(center_buf)
                        total_pairs_processed += len(center_buf)

                        # Learning rate decay (linear)
                        progress = total_pairs_processed / (epochs * estimated_total_pairs)
                        self.learning_rate = max(0.0001, initial_lr * (1 - progress))

                        center_buf.clear()
                        context_buf.clear()

                        # Progress reporting
                        if verbose and pair_count - last_report >= report_interval:
                            avg_loss_so_far = epoch_loss / pair_count
                            print(f"  Epoch {epoch + 1}/{epochs} - Pairs: {pair_count:,}, "
                                  f"Loss: {avg_loss_so_far:.4f}, LR: {self.learning_rate:.6f}")
                            last_report = pair_count

                        if max_pairs_per_epoch and pair_count >= max_pairs_per_epoch:
                            break

                if max_pairs_per_epoch and pair_count >= max_pairs_per_epoch:
                    break

            # Process remaining pairs
            if center_buf:
                loss = self.train_batch(
                    np.array(center_buf, dtype=np.int32),
                    np.array(context_buf, dtype=np.int32)
                )
                epoch_loss += loss
                pair_count += len(center_buf)
                total_pairs_processed += len(center_buf)

            avg_loss = epoch_loss / pair_count if pair_count > 0 else 0
            self.loss_history.append(avg_loss)

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, "
                      f"Pairs: {pair_count:,}, LR: {self.learning_rate:.6f}")
    
    def get_word_vector(self, word_id: int) -> np.ndarray:
        """Get the embedding vector for a word."""
        return self.W_in[word_id]
    
    def get_all_vectors(self) -> np.ndarray:
        """Get all word embeddings (input vectors)."""
        return self.W_in
    
    def save(self, filepath: str):
        """Save the model to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'W_in': self.W_in,
                'W_out': self.W_out,
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim,
                'window_size': self.window_size,
                'negative_samples': self.negative_samples,
                'loss_history': self.loss_history
            }, f)
    
    @classmethod
    def load(cls, filepath: str):
        """Load a model from a file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(
            vocab_size=data['vocab_size'],
            embedding_dim=data['embedding_dim'],
            window_size=data['window_size'],
            negative_samples=data['negative_samples']
        )
        model.W_in = data['W_in']
        model.W_out = data['W_out']
        model.loss_history = data['loss_history']
        
        return model


class Word2VecCBOW(Word2Vec):
    """
    CBOW (Continuous Bag of Words) variant of Word2Vec.
    
    Instead of predicting context from center word (skip-gram),
    CBOW predicts center word from context words.
    
    The context vectors are averaged before prediction.
    """
    
    def generate_training_pairs(
        self,
        word_ids: List[int]
    ) -> List[Tuple[List[int], int]]:
        """
        Generate (context_words, center_word) pairs.
        
        Returns list of (context_word_ids, center_word_id) tuples.
        """
        pairs = []
        sentence_length = len(word_ids)
        
        for center_pos in range(sentence_length):
            center_word = word_ids[center_pos]
            
            # Define context window
            window = np.random.randint(1, self.window_size + 1)
            
            # Get context words
            start = max(0, center_pos - window)
            end = min(sentence_length, center_pos + window + 1)
            
            context_words = []
            for context_pos in range(start, end):
                if context_pos != center_pos:
                    context_words.append(word_ids[context_pos])
            
            if context_words:
                pairs.append((context_words, center_word))
        
        return pairs
    
    def forward(self, context_words: List[int], center_word: int, negative_words: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass for CBOW.
        
        Context word vectors are averaged to form the input.
        """
        # Average context word vectors
        context_vecs = self.W_in[context_words]  # Shape: (num_context, embedding_dim)
        context_vec = np.mean(context_vecs, axis=0)  # Shape: (embedding_dim,)
        
        # Get output vector for center word
        center_out = self.W_out[center_word]  # Shape: (embedding_dim,)
        
        # Get output vectors for negative samples
        negative_out = self.W_out[negative_words]  # Shape: (negative_samples, embedding_dim)
        
        return context_vec, center_out, negative_out
    
    def backward(
        self,
        context_words: List[int],
        center_word: int,
        negative_words: np.ndarray,
        context_vec: np.ndarray,
        center_out: np.ndarray,
        negative_out: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass for CBOW.
        
        Gradient is distributed equally to all context words.
        """
        num_context = len(context_words)
        
        # Positive sample gradients
        pos_score = np.dot(center_out, context_vec)
        pos_sigmoid = self._sigmoid(pos_score)
        pos_grad_coef = pos_sigmoid - 1
        
        # Gradient for context vector (will be distributed to context words)
        grad_context_vec = pos_grad_coef * center_out
        
        # Gradient for center output vector
        grad_center = pos_grad_coef * context_vec
        
        # Negative samples gradients
        neg_scores = np.dot(negative_out, context_vec)
        neg_sigmoids = self._sigmoid(neg_scores)
        neg_grad_coefs = neg_sigmoids
        
        # Add gradient from negative samples to context vector
        grad_context_vec += np.dot(neg_grad_coefs, negative_out)
        
        # Gradients for negative output vectors
        grad_negative = np.outer(neg_grad_coefs, context_vec)
        
        # Distribute gradient equally to all context words
        grad_context = np.tile(grad_context_vec / num_context, (num_context, 1))
        
        return grad_context, grad_center, grad_negative
    
    def update_parameters(
        self,
        context_words: List[int],
        center_word: int,
        negative_words: np.ndarray,
        grad_context: np.ndarray,
        grad_center: np.ndarray,
        grad_negative: np.ndarray
    ):
        """Update parameters for CBOW."""
        # Update context word input vectors
        for i, word_id in enumerate(context_words):
            self.W_in[word_id] -= self.learning_rate * grad_context[i]
        
        # Update center word output vector
        self.W_out[center_word] -= self.learning_rate * grad_center
        
        # Update negative word output vectors
        for i, neg_word in enumerate(negative_words):
            self.W_out[neg_word] -= self.learning_rate * grad_negative[i]
    
    def train_step(self, context_words: List[int], center_word: int) -> float:
        """Training step for CBOW."""
        # Get negative samples
        negative_words = self.get_negative_samples(
            self.negative_samples,
            np.array([center_word])
        )[0]

        # Forward pass
        context_vec, center_out, negative_out = self.forward(
            context_words, center_word, negative_words
        )

        # Compute loss
        loss = self.compute_loss(context_vec, center_out, negative_out)

        # Backward pass
        grad_context, grad_center, grad_negative = self.backward(
            context_words, center_word, negative_words,
            context_vec, center_out, negative_out
        )

        # Update parameters
        self.update_parameters(
            context_words, center_word, negative_words,
            grad_context, grad_center, grad_negative
        )

        return loss

    def train(
        self,
        sentences: List[List[int]],
        epochs: int = 5,
        verbose: bool = True,
        max_pairs_per_epoch: int = None,
        report_interval: int = 100000,
        batch_size: int = 64
    ):
        """Train CBOW model (per-pair, since context lengths vary)."""
        total_words = sum(len(s) for s in sentences)
        estimated_total_pairs = total_words * self.window_size * 2
        total_pairs_processed = 0
        initial_lr = self.learning_rate

        for epoch in range(epochs):
            epoch_loss = 0.0
            pair_count = 0
            last_report = 0

            for sentence in sentences:
                subsampled = self.subsample_words(sentence)
                pairs = self.generate_training_pairs(subsampled)

                for context_words, center_word in pairs:
                    loss = self.train_step(context_words, center_word)
                    epoch_loss += loss
                    pair_count += 1
                    total_pairs_processed += 1

                    progress = total_pairs_processed / (epochs * estimated_total_pairs)
                    self.learning_rate = max(0.0001, initial_lr * (1 - progress))

                    if verbose and pair_count - last_report >= report_interval:
                        avg_loss_so_far = epoch_loss / pair_count
                        print(f"  Epoch {epoch + 1}/{epochs} - Pairs: {pair_count:,}, "
                              f"Loss: {avg_loss_so_far:.4f}, LR: {self.learning_rate:.6f}")
                        last_report = pair_count

                    if max_pairs_per_epoch and pair_count >= max_pairs_per_epoch:
                        break

                if max_pairs_per_epoch and pair_count >= max_pairs_per_epoch:
                    break

            avg_loss = epoch_loss / pair_count if pair_count > 0 else 0
            self.loss_history.append(avg_loss)

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, "
                      f"Pairs: {pair_count:,}, LR: {self.learning_rate:.6f}")
