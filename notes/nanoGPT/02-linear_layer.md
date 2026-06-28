# nn.Linear

## Purpose

`nn.Linear` is one of the most fundamental building blocks in deep learning.

It performs a **linear transformation** on the input tensor by multiplying it with a learnable weight matrix and optionally adding a learnable bias.

Mathematically:

```
Output = Input × Weightᵀ + Bias
```

During training, the weights and biases are updated so that the model learns useful representations from the data.

---

# Definition

```python
nn.Linear(in_features, out_features)
```

Where:

- **in_features** = Size of the input feature vector.
- **out_features** = Size of the output feature vector.

Example:

```python
layer = nn.Linear(768, 2304)
```

This means:

- Every input vector has **768 features**
- Every output vector will have **2304 features**

---

# Input Shape

Suppose the input tensor is:

```python
x.shape = (32, 128, 768)
```

Where:

- **32** → Batch Size
- **128** → Number of Tokens (Sequence Length)
- **768** → Embedding Dimension

The Linear layer only operates on the **last dimension**.

---

# Output Shape

Applying

```python
layer = nn.Linear(768, 2304)

y = layer(x)
```

gives

```python
y.shape = (32, 128, 2304)
```

Notice:

- Batch Size remains unchanged.
- Sequence Length remains unchanged.
- Only the embedding dimension changes.

---

# Why only the last dimension changes?

A Linear layer treats every feature vector independently.

For

```python
x.shape = (32,128,768)
```

there are

```
32 × 128 = 4096
```

individual vectors.

Each vector has length

```
768
```

Every one of these vectors is transformed independently from

```
768 → 2304
```

The batch and token dimensions simply organize the data and are **not mixed together**.

---

# Weight Matrix

For

```python
nn.Linear(768,2304)
```

the learnable weight matrix has shape

```python
(2304,768)
```

Notice:

```
Weight Shape

(out_features, in_features)
```

NOT

```
(in_features, out_features)
```

This is because PyTorch internally computes

```
Output = Input × Weightᵀ
```

---

# Bias

By default,

```python
nn.Linear(...)
```

also learns a bias vector.

Shape:

```python
Bias.shape = (2304,)
```

The same bias vector is added to every input vector.

Mathematically:

```
Output = Input × Weightᵀ + Bias
```

---

# Example

Suppose

```python
x.shape = (2,5,4)
```

Meaning:

- Batch = 2
- Tokens = 5
- Features = 4

Apply

```python
layer = nn.Linear(4,8)
```

Output:

```python
y.shape = (2,5,8)
```

Only the last dimension changes.

---

# Why is nn.Linear used in Transformers?

Transformers frequently need to change the representation size.

Examples:

## 1. Creating Query, Key and Value

```python
self.c_attn = nn.Linear(768,2304)
```

Output:

```python
(B,T,768)

↓

(B,T,2304)
```

Later split into:

```python
Q = (B,T,768)

K = (B,T,768)

V = (B,T,768)
```

---

## 2. Feed Forward Network (MLP)

Inside every Transformer block:

```python
768

↓

3072

↓

768
```

implemented as

```python
nn.Linear(768,3072)

nn.Linear(3072,768)
```

This allows the model to learn richer feature representations.

---

# Does nn.Linear process all tokens together?

No.

Each token is processed independently.

Example:

Input:

```python
(B,T,C)
```

becomes conceptually

```
Token 1 → Linear

Token 2 → Linear

Token 3 → Linear

...

Token T → Linear
```

The same weight matrix is shared across every token.

This sharing allows the model to generalize across different sequence lengths.

---

# Shape Evolution Example

Input

```python
(32,128,768)
```

↓

Linear(768 → 2304)

↓

```python
(32,128,2304)
```

Only the final dimension changes.

---

# Key Takeaways

- `nn.Linear` performs a learnable linear transformation.
- It changes only the last dimension of the tensor.
- Batch size and sequence length remain unchanged.
- The weight matrix has shape `(out_features, in_features)`.
- The same weights are shared across all tokens.
- In GPT, `nn.Linear` is used to generate Query, Key, Value vectors and inside the Feed Forward Network (MLP).
