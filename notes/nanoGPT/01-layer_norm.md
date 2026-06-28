# LayerNorm (`model.py`)

## Purpose

Layer Normalization stabilizes training by normalizing the embedding vector of **each token independently**. After normalization, it applies learnable scaling (`γ`) and shifting (`β`) so the model can recover useful representations if needed.

Unlike BatchNorm, LayerNorm does not depend on other samples in the batch.

---

## Input

```python
x.shape = (B, T, C)
```

Where:

- **B** = Batch Size
- **T** = Sequence Length (number of tokens)
- **C** = Embedding Dimension

Example:

```python
x.shape = (32, 128, 768)
```

---

## Learnable Parameters

```python
self.weight = nn.Parameter(torch.ones(C))
self.bias   = nn.Parameter(torch.zeros(C))
```

### Shape

```python
weight.shape = (C,)
bias.shape   = (C,)
```

For GPT-2 Small:

```python
weight.shape = (768,)
```

Each embedding dimension has its own learnable scale (`γ`) and bias (`β`).

---

## Forward Pass

Internally, `F.layer_norm()` approximately performs:

```python
mean = x.mean(dim=-1, keepdim=True)

variance = x.var(dim=-1, keepdim=True)

normalized = (x - mean) / torch.sqrt(variance + eps)

output = normalized * weight + bias
```

---

## Shape Transformations

Input:

```python
(B,T,C)
```

### Step 1

Compute mean

```python
mean = x.mean(dim=-1, keepdim=True)
```

Shape:

```python
(B,T,1)
```

---

### Step 2

Center values

```python
centered = x - mean
```

Broadcasting expands

```python
(B,T,1)
```

to

```python
(B,T,C)
```

Result:

```python
(B,T,C)
```

---

### Step 3

Normalize

```python
normalized = centered / std
```

Shape remains

```python
(B,T,C)
```

---

### Step 4

Scale and shift

```python
output = normalized * γ + β
```

Shapes

```python
γ = (C,)
β = (C,)
```

Broadcast automatically to

```python
(B,T,C)
```

Output

```python
(B,T,C)
```

---

## Broadcasting Insight

Broadcasting allows tensors with fewer dimensions to behave like larger tensors during arithmetic.

Example:

```python
(32,128,768)

-

(32,128,1)

↓

(32,128,768)
```

The mean is automatically copied across all embedding dimensions.

---

## Key Takeaways

- LayerNorm normalizes **one token embedding at a time**.
- It computes statistics across the **embedding dimension (`C`)**.
- It does **not** normalize across batches or tokens.
- Output shape is identical to input shape.
- `γ` and `β` are learnable parameters with one value per embedding dimension.
- `eps = 1e-5` prevents division by zero.

## Common Misconceptions

LayerNorm normalizes the **entire embedding vector** of one token and not every embedding dimension independently.

---

Only the intermediate mean/variance tensors are reduced.
The final output retains the original shape.

---

Each token is normalized independently to prevent the information mixing.

---

γ and β i.e weight and bias respt. are vectors and not metrices with one value per embedding dimension.

---
