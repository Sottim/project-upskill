# Residual Connections (Skip Connections)

## Purpose

A Residual Connection allows the model to preserve the original information while adding newly learned information.

Instead of replacing the input completely, the Transformer learns **what should be added** to the existing representation.

---

# Motivation

Without a Residual Connection, every Transformer block would completely overwrite the input.

```
Input

↓

Attention

↓

MLP

↓

Output
```

As the input passes through many layers, the original representation gradually disappears.

With a Residual Connection, every block keeps the original input and only adds useful changes.

```
Output = Input + Learned Information
```

---

# Implementation in nanoGPT

```python
x = x + self.attn(self.ln_1(x))
x = x + self.mlp(self.ln_2(x))
```

Both the Attention block and the MLP learn a **correction** to the input instead of creating a completely new representation.

---

# Why is this useful?

Suppose the input embedding is

```
[2, 5, 1]
```

Attention learns

```
[0.3, -0.5, 2]
```

The output becomes

```
[2.3, 4.5, 3]
```

The original information is still present while new contextual information is added.

---

# Learning the Residual

Instead of learning

```
Output
```

the network learns

```
Output − Input
```

This difference is called the **Residual**.

Learning a small correction is usually much easier than learning an entirely new representation from scratch.

---

# Benefits

- Preserves the original representation.
- Makes training deep networks much more stable.
- Allows every layer to make small improvements instead of large changes.
- If a layer is not useful, it can learn to contribute very little, effectively behaving like an identity function.

---

# Shape

Residual addition requires both tensors to have the same shape.

Example:

```python
Input               (32,128,768)

Attention Output    (32,128,768)

↓

Addition

↓

Output              (32,128,768)
```

This is one reason why both the Attention block and the MLP project their outputs back to the original embedding dimension.

---

# Transformer Block

```
Input
   │
LayerNorm
   │
Attention
   │
+ Residual
   │
LayerNorm
   │
MLP
   │
+ Residual
   │
Output
```

---

# Key Takeaways

- Residual Connections preserve the original information while adding newly learned information.
- The network learns **what to add**, not **what to replace**.
- They make training deep Transformer models stable.
- Every Transformer block contains two Residual Connections: one after Attention and one after the MLP.
