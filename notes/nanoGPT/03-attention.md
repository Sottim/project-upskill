# Attention (`CausalSelfAttention`)

## Purpose

Self-Attention allows every token to gather relevant information from other tokens in the sequence.

Unlike traditional neural networks that process tokens independently, Self-Attention creates **context-aware embeddings** by allowing each token to "look at" other tokens before generating its new representation.

Example:

```
"The animal didn't cross the street because it was tired."
```

The token **"it"** should understand that it refers to **"animal"**, not **"street"**. Self-Attention learns these relationships automatically during training.

---

# Input

```python
x.shape = (B, T, C)
```

Where:

- **B** = Batch Size
- **T** = Sequence Length (Number of Tokens)
- **C** = Embedding Dimension

Example:

```python
x.shape = (32, 128, 768)
```

---

# Step 1: Generate Query, Key and Value

Instead of using the embedding directly, GPT creates three different representations of every token.

```python
self.c_attn = nn.Linear(C, 3 * C)
```

Example:

```python
nn.Linear(768, 2304)
```

Input:

```python
(B,T,768)
```

Output:

```python
(B,T,2304)
```

The output is then split into three equal parts.

```python
q, k, v = qkv.split(C, dim=2)
```

Shapes:

```python
q.shape = (B,T,C)

k.shape = (B,T,C)

v.shape = (B,T,C)
```

---

# What are Query, Key and Value?

Every token creates three different vectors.

### Query (Q)

Represents:

> "What information am I looking for?"

---

### Key (K)

Represents:

> "What information do I contain?"

The similarity between Query and Key determines **which tokens are important**.

---

### Value (V)

Represents:

> "What information should I contribute if another token attends to me?"

The Value vector contains the actual information transferred during attention.

---

# Step 2: Multi-Head Attention

Instead of performing one large attention operation over the entire embedding, GPT splits the embedding into multiple heads.

Example:

```python
Embedding Size = 768

Heads = 12

Head Size = 64
```

Since

```
768 / 12 = 64
```

Every head learns different relationships within the sentence.

Example specializations (learned automatically):

- Grammar
- Long-range dependencies
- Pronoun resolution
- Semantic relationships

---

# Tensor Reshaping

Before splitting:

```python
q.shape = (B,T,768)
```

After

```python
q.view(B,T,12,64)
```

Shape:

```python
(B,T,12,64)
```

After

```python
q.transpose(1,2)
```

Shape:

```python
(B,12,T,64)
```

Dimensions:

- Batch
- Attention Heads
- Tokens
- Head Features

The same transformation is applied to Key and Value.

---

# Step 3: Compute Attention Scores

```python
att = q @ k.transpose(-2,-1)
```

Shapes:

```
Q

(B,H,T,D)

Kᵀ

(B,H,D,T)
```

Matrix Multiplication:

```
(T,D)

@

(D,T)

↓

(T,T)
```

Final Shape:

```python
att.shape = (B,H,T,T)
```

---

# What does the Attention Matrix represent?

Rows represent:

> Query Tokens

Columns represent:

> Key Tokens

Every element represents:

> "How much should this token attend to another token?"

Example:

```
            Keys

          I  love deep learning

Query I

Query love

Query deep

Query learning
```

Each value is a similarity score computed using the dot product between Query and Key vectors.

---

# Step 4: Scale Attention Scores

```python
att = att * (1 / sqrt(head_size))
```

Purpose:

Large dot products cause Softmax to become extremely peaked, making training unstable.

Scaling keeps attention scores within a reasonable numerical range.

Example:

```
head_size = 64

sqrt(64) = 8

scaled_score = score / 8
```

---

# Step 5: Apply Causal Mask

GPT should never look into future tokens while predicting the next word.

A lower triangular mask is created.

Example for 5 tokens:

```
1 0 0 0 0

1 1 0 0 0

1 1 1 0 0

1 1 1 1 0

1 1 1 1 1
```

The mask is applied as:

```python
att = att.masked_fill(mask == 0, -inf)
```

Future positions receive:

```
-∞
```

After Softmax:

```
exp(-∞) = 0
```

Meaning future tokens receive exactly **zero attention**.

---

# Why is the diagonal equal to 1?

Every token is allowed to attend to itself.

A token's own embedding already contains useful information and should be included when computing its contextual representation.

---

# Step 6: Softmax

```python
att = F.softmax(att, dim=-1)
```

Softmax converts similarity scores into probabilities.

Properties:

- Values lie between 0 and 1
- Every row sums to 1
- Larger similarity scores receive larger probabilities

These probabilities are called **Attention Weights**.

---

# Step 7: Weighted Sum of Values

```python
y = att @ v
```

Shapes:

```
Attention

(B,H,T,T)

@

Value

(B,H,T,D)

↓

(B,H,T,D)
```

Every row of the attention matrix is used as weights to combine Value vectors.

Example:

```
Attention Weights

animal      0.80

street      0.10

because     0.05

cross       0.05
```

The new embedding becomes:

```
0.80 × Value(animal)

+

0.10 × Value(street)

+

0.05 × Value(because)

+

0.05 × Value(cross)
```

Instead of copying one token's information, GPT creates a weighted combination of multiple Value vectors.

---

# Complete Attention Pipeline

```
Input Embeddings

        │

        ▼

Linear Layer

        │

        ▼

Generate Query, Key, Value

        │

        ▼

Split into Multiple Heads

        │

        ▼

Q × Kᵀ

        │

        ▼

Similarity Scores

        │

        ▼

Scale by √head_size

        │

        ▼

Apply Causal Mask

        │

        ▼

Softmax

        │

        ▼

Attention Weights

        │

        ▼

Weighted Sum of Values

        │

        ▼

Context-Aware Embeddings
```

---

# Shape Evolution

```
Input

(B,T,C)

↓

Linear

(B,T,3C)

↓

Split

Q,K,V

(B,T,C)

↓

View

(B,T,H,D)

↓

Transpose

(B,H,T,D)

↓

Q @ Kᵀ

(B,H,T,T)

↓

Softmax

(B,H,T,T)

↓

att @ V

(B,H,T,D)
```

---

# Key Ideas

---

Query-Key similarity determines importance. Value contains the information that is transferred.

---

Attention creates a weighted combination of multiple Value vectors.

---

Causal masking prevents attention to future positions.

---

# Key Takeaways

- Every token generates Query, Key and Value vectors.
- Query-Key dot products compute similarity scores.
- Softmax converts similarities into attention probabilities.
- Value vectors contain the information transferred between tokens.
- Multi-Head Attention allows different heads to learn different relationships.
- Causal masking ensures GPT only attends to previous tokens and itself unlike BERT models
- Self-Attention transforms static token embeddings into context-aware embeddings.
