# Word2Vec in Pure NumPy

A clean, educational implementation of Word2Vec (Skip-gram and CBOW) with Negative Sampling, implemented entirely in NumPy without any deep learning frameworks.

## Overview

This repository contains a from-scratch implementation of the Word2Vec algorithm as described in:
> Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). [Distributed representations of words and phrases and their compositionality](https://arxiv.org/abs/1310.4546). Advances in neural information processing systems.

### Features

- **Skip-gram Model**: Predict context words given a center word
- **CBOW Model**: Predict center word given context words (averaged)
- **Negative Sampling**: Efficient approximation of softmax
- **Subsampling**: Reduce impact of frequent words
- **Full Gradient Derivation**: All gradients computed manually
- **Clean Architecture**: Well-documented, modular code

## Mathematical Foundation

### Skip-gram Objective

Given a center word $w_c$ and context word $w_o$, the objective is to maximize:

$$J(\theta) = \log \sigma(u_o^T v_c) + \sum_{i=1}^{k} \log \sigma(-u_{n_i}^T v_c)$$

Where:
- $v_c \in \mathbb{R}^d$: input vector for center word (from matrix $W_{in}$)
- $u_o \in \mathbb{R}^d$: output vector for positive context word (from matrix $W_{out}$)
- $u_{n_i} \in \mathbb{R}^d$: output vectors for $k$ negative samples
- $\sigma(x) = \frac{1}{1 + e^{-x}}$: sigmoid function

### Loss Function

The negative log-likelihood loss for a single training pair:

$$\mathcal{L} = -\log \sigma(u_o^T v_c) - \sum_{i=1}^{k} \log \sigma(-u_{n_i}^T v_c)$$

### Gradient Derivation

#### Positive Sample Gradients

Let $z_{pos} = u_o^T v_c$ (the score for the positive sample).

**Chain rule setup:**

$$\frac{\partial \mathcal{L}}{\partial z_{pos}} = \frac{\partial}{\partial z_{pos}} [-\log \sigma(z_{pos})] = -\frac{1}{\sigma(z_{pos})} \cdot \sigma(z_{pos})(1 - \sigma(z_{pos})) = \sigma(z_{pos}) - 1$$

**Gradients for vectors:**

$$\frac{\partial \mathcal{L}}{\partial v_c} = \frac{\partial \mathcal{L}}{\partial z_{pos}} \cdot \frac{\partial z_{pos}}{\partial v_c} = (\sigma(z_{pos}) - 1) \cdot u_o$$

$$\frac{\partial \mathcal{L}}{\partial u_o} = \frac{\partial \mathcal{L}}{\partial z_{pos}} \cdot \frac{\partial z_{pos}}{\partial u_o} = (\sigma(z_{pos}) - 1) \cdot v_c$$

#### Negative Sample Gradients

For each negative sample $n_i$, let $z_{neg_i} = u_{n_i}^T v_c$.

**Chain rule setup:**

$$\frac{\partial \mathcal{L}}{\partial z_{neg_i}} = \frac{\partial}{\partial z_{neg_i}} [-\log \sigma(-z_{neg_i})] = -\frac{1}{\sigma(-z_{neg_i})} \cdot \sigma(-z_{neg_i})(1 - \sigma(-z_{neg_i})) \cdot (-1) = \sigma(z_{neg_i})$$

**Gradients for vectors:**

$$\frac{\partial \mathcal{L}}{\partial v_c} \mathrel{+}= \sigma(z_{neg_i}) \cdot u_{n_i}$$

$$\frac{\partial \mathcal{L}}{\partial u_{n_i}} = \sigma(z_{neg_i}) \cdot v_c$$

### Combined Gradient for Center Word

$$\frac{\partial \mathcal{L}}{\partial v_c} = (\sigma(u_o^T v_c) - 1) \cdot u_o + \sum_{i=1}^{k} \sigma(u_{n_i}^T v_c) \cdot u_{n_i}$$

### Parameter Updates (SGD)

$$v_c \leftarrow v_c - \eta \cdot \frac{\partial \mathcal{L}}{\partial v_c}$$

$$u_o \leftarrow u_o - \eta \cdot \frac{\partial \mathcal{L}}{\partial u_o}$$

$$u_{n_i} \leftarrow u_{n_i} - \eta \cdot \frac{\partial \mathcal{L}}{\partial u_{n_i}}$$

## CBOW Variant

CBOW (Continuous Bag of Words) predicts the center word from context words.

### Forward Pass

Context vectors are averaged:

$$v_{context} = \frac{1}{|C|} \sum_{w \in C} v_w$$

Where $C$ is the set of context words.

### Gradients

The gradient with respect to the context vector is distributed equally to all context word vectors:

$$\frac{\partial \mathcal{L}}{\partial v_w} = \frac{1}{|C|} \cdot \frac{\partial \mathcal{L}}{\partial v_{context}} \quad \text{for all } w \in C$$

## Negative Sampling

Instead of computing softmax over the entire vocabulary $V$ (expensive: $O(|V|)$), we sample $k$ negative words and approximate:

$$P(w_o | w_c) \approx \sigma(u_o^T v_c) \prod_{i=1}^{k} \sigma(-u_{n_i}^T v_c)$$

### Sampling Distribution

Words are sampled from a modified unigram distribution:

$$P(w) = \frac{f(w)^{0.75}}{\sum_{w'} f(w')^{0.75}}$$

Where $f(w)$ is the word frequency. The $0.75$ power smooths the distribution, giving rare words slightly higher probability.

## Subsampling

Frequent words ("the", "a", "is") are subsampled to improve training:

$$P(\text{discard } w) = 1 - \sqrt{\frac{t}{f(w)}}$$

Where $t$ is a threshold (typically $10^{-5}$) and $f(w)$ is the word frequency.

## Project Structure

```
word2vec-numpy/
├── word2vec.py          # Core implementation (Word2Vec, Word2VecCBOW classes)
├── preprocessing.py     # Text preprocessing and vocabulary building
├── train.py            # Training script with evaluation
├── README.md           # This file
└── requirements.txt    # Dependencies
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/word2vec-numpy.git
cd word2vec-numpy

# Install dependencies
pip install numpy
```

## Usage

### Quick Start (Sample Text)

```bash
python train.py --data sample --epochs 1000
```

### Full Training (Text8 Dataset)

```bash
python train.py --data text8 --epochs 5 --embedding-dim 100
```

### Custom Parameters

```bash
python train.py \
    --data text8 \
    --model-type skipgram \
    --embedding-dim 200 \
    --window-size 8 \
    --negative-samples 10 \
    --learning-rate 0.025 \
    --epochs 5 \
    --min-count 5
```

### Programmatic Usage

```python
from word2vec import Word2Vec
from preprocessing import preprocess_sentences, split_into_sentences

# Prepare data
texts = ["machine learning is fascinating", "deep learning models are powerful"]
sentences_text = split_into_sentences(' '.join(texts))
vocab, sentences = preprocess_sentences(sentences_text, min_count=1)

# Initialize model
model = Word2Vec(
    vocab_size=len(vocab),
    embedding_dim=100,
    window_size=5,
    negative_samples=5,
    learning_rate=0.025
)

# Build negative sampling table
word_counts = vocab.get_word_counts()
model.build_negative_sampling_table(word_counts)

# Train
model.train(sentences, epochs=10, verbose=True)

# Get word vector
vector = model.get_word_vector(vocab.word2id['learning'])

# Save model
model.save('word2vec_model.pkl')
```

## Implementation Details

### Weight Initialization

- **Input embeddings ($W_{in}$)**: Uniform distribution $\mathcal{U}(-\frac{0.5}{d}, \frac{0.5}{d})$
- **Output embeddings ($W_{out}$)**: Zeros (common practice)

### Numerical Stability

The sigmoid function is clipped to avoid overflow:

```python
def sigmoid(x):
    x = np.clip(x, -6, 6)  # Prevent overflow
    return 1 / (1 + np.exp(-x))
```

### Learning Rate Decay

Learning rate decays linearly during training:

$$\eta_t = \eta_0 \cdot \left(1 - \frac{t}{T}\right)$$

Where $T$ is the total number of training steps.

## Evaluation

The training script includes two evaluation tasks:

### 1. Similar Words

Finds the most similar words using cosine similarity:

```python
similar = find_similar_words(model, vocab, 'learning', top_k=5)
```

### 2. Analogies

Solves analogies of the form: $a$ is to $b$ as $c$ is to $?$ (vector arithmetic: $b - a + c$)

```python
results = analogies(model, vocab, 'king', 'man', 'woman')
# king - man + woman ≈ queen
```

## Algorithm Complexity

| Operation | Complexity |
|-----------|------------|
| Forward pass | $O(d \cdot (1 + k))$ |
| Backward pass | $O(d \cdot (1 + k))$ |
| Full softmax (for comparison) | $O(d \cdot \|V\|)$ |

Where:
- $d$: embedding dimension
- $k$: number of negative samples
- $\|V\|$: vocabulary size

## Differences from Original Word2Vec

1. **No hierarchical softmax**: We use negative sampling only
2. **Fixed learning rate decay**: Original uses linear decay; we implement similar
3. **No multi-threading**: Single-threaded for clarity
4. **Pure NumPy**: No C optimizations (slower but more readable)

## Possible Extensions

1. **Hierarchical Softmax**: Alternative to negative sampling
2. **Mini-batch Training**: Current implementation uses SGD (batch_size=1)
3. **Adaptive Learning Rates**: AdaGrad, Adam
4. **Phrase Detection**: Detect multi-word expressions
5. **Subword Information**: Character n-grams (FastText-style)

## Testing

Run a quick test on sample data:

```bash
python train.py --data sample --epochs 1000
```

Expected output shows decreasing loss and meaningful similar words.

## References

1. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). [Efficient estimation of word representations in vector space](https://arxiv.org/abs/1301.3781). arXiv preprint arXiv:1301.3781.

2. Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). [Distributed representations of words and phrases and their compositionality](https://arxiv.org/abs/1310.4546). Advances in neural information processing systems.

3. Rong, X. (2014). [word2vec parameter learning explained](https://arxiv.org/abs/1411.2738). arXiv preprint arXiv:1411.2738.

## License

MIT License - Feel free to use for educational and research purposes.

## Contributing

This is an educational implementation. Contributions that improve clarity or add educational value are welcome!
