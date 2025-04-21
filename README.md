# NdLinear for Transformer Architecture

## About
This project implements and explores the application of NdLinear specifically for transformer models. NdLinear is a parameter-efficient alternative to standard linear layers that significantly improves transformer model performance. The original NdLinear module is developed by [Ensemble AI](https://ensemblecore.ai) and is available at [github.com/ensemble-core/NdLinear](https://github.com/ensemble-core/NdLinear).

## Why NdLinear for Transformers?

Transformer models rely heavily on linear projections in attention mechanisms, feed-forward networks, and embedding layers. By replacing standard nn.Linear layers with NdLinear, we can:

1. **Reduce Parameter Count**: NdLinear significantly decreases the number of parameters in transformer models while maintaining or improving performance.

2. **Preserve Multi-dimensional Structure**: Unlike standard linear layers that flatten input tensors, NdLinear preserves the multi-dimensional structure of attention matrices and token embeddings.

3. **Improve Training Efficiency**: Transformer models using NdLinear typically train faster and converge more quickly than those using standard linear layers.

4. **Enhance Model Performance**: By capturing multivariate structure and dependencies typically lost in standard fully connected layers, NdLinear improves overall transformer model performance.

## Installation

To integrate NdLinear into your transformer projects:

```bash
git clone https://github.com/ensemble-core/NdLinear.git
cd NdLinear
pip install . 
```

Alternatively, if packaged, install via pip:

```bash
pip install ndlinear
```

Or, via conda:

```bash
conda install conda-forge::ndlinear
```

## Implementation in Transformer Models

### Replacing Attention Projection Layers

```python
import torch
from ndlinear import NdLinear
import torch.nn as nn

class NdLinearAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Replace standard linear projections with NdLinear
        self.q_proj = NdLinear(input_dims=(embed_dim,), hidden_size=(embed_dim,))
        self.k_proj = NdLinear(input_dims=(embed_dim,), hidden_size=(embed_dim,))
        self.v_proj = NdLinear(input_dims=(embed_dim,), hidden_size=(embed_dim,))
        self.out_proj = NdLinear(input_dims=(embed_dim,), hidden_size=(embed_dim,))
        
    def forward(self, query, key, value):
        batch_size = query.shape[0]
        
        # Linear projections using NdLinear
        q = self.q_proj(query).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention calculations (same as standard transformer)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output using NdLinear
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)
        output = self.out_proj(attn_output)
        
        return output
```

### Replacing Feed-Forward Networks

```python
import torch
import torch.nn as nn
from ndlinear import NdLinear

class NdLinearFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # Replace standard linear layers with NdLinear
        self.linear1 = NdLinear(input_dims=(d_model,), hidden_size=(d_ff,))
        self.linear2 = NdLinear(input_dims=(d_ff,), hidden_size=(d_model,))
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
```

### Complete Transformer Block with NdLinear

```python
import torch
import torch.nn as nn
from ndlinear import NdLinear

class NdLinearTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = NdLinearAttention(embed_dim, num_heads)
        self.ff = NdLinearFeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention with residual connection and normalization
        attn_output = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and normalization
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```


## Advanced Usage: Multi-dimensional Attention

NdLinear enables more sophisticated attention mechanisms by preserving tensor structure:

```python
import torch
from ndlinear import NdLinear

# Input with shape: (batch_size, seq_len, embed_dim)
input_tensor = torch.randn(32, 128, 768)

# Reshape to preserve structural information
reshaped_input = input_tensor.reshape(32, 128, 768, 1)

# Apply NdLinear while preserving dimensions
ndlinear_layer = NdLinear(input_dims=(768, 1), hidden_size=(768, 4))
output = ndlinear_layer(reshaped_input)

# Reshape back for subsequent processing
output = output.reshape(32, 128, 768, 4).mean(dim=-1)
```

## References

This project uses NdLinear from the official repository:
- [github.com/ensemble-core/NdLinear](https://github.com/ensemble-core/NdLinear)

For more information and community support, visit:
- [Ensemble AI](https://ensemblecore.ai)
- [Discord Community](https://discord.gg/6DWHusWN)

## License

The original NdLinear project is distributed under the Apache 2.0 license. Please refer to the [original repository](https://github.com/ensemble-core/NdLinear) for license details.
