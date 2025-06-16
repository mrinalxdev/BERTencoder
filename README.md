# ON MY WAY TO WRITE A BERT ENCODER

Bi-directional encoder representations from transformers, here are the things I need to cover for building one
- [x] Implement the multi-head self-attention mechanism
    - the input is given to queries, keys, and values with linear transformation
    - splitting the heads and compute scaled dot product attention for each head
    - Concatenate the heads and project back

- [x] Implement the position-wise feed-forward network. 
    - Two Linear transformations with GELU activation in between
- [ ] Implement layer normalization and residual connections
- [ ] Combining these into a single encoder layer
- [ ] Stacking multiple encoder layers to form the BERT encoder