# ON MY WAY TO WRITE A BERT ENCODER

Bi-directional encoder representations from transformers, here are the things I need to cover for building one
- [x] Implement the multi-head self-attention mechanism
    - the input is given to queries, keys, and values with linear transformation
    - splitting the heads and compute scaled dot product attention for each head
    - Concatenate the heads and project back

- [x] Implement the position-wise feed-forward network. 
    - Two Linear transformations with GELU activation in between

Math behind the `TransformerEncoderLayer` :
    - Attention sub-layer : $ x' = LayerNorm(x + Dropout(Attention(x))) $

- [x] Implement layer normalization and residual connections
    - stack of `num_hidden_layers` encoder layers
- [x] Combining these into a single encoder layer
- [x] Stacking multiple encoder layers to form the BERT encoder


## Here's what I am doing next [Visualizing Attention Heads]

Extracting the attention_probs from MultiHeadAttention module and create a heatmap to show how tokens attend to each other

My Approach

- Take the first sequence in the batch and then plot the heatmap

How is visualization part is working ?


## Experimenting next with text classification

Approach : 

1. Classification Head (BERTClassifier) : the `BERTClassifier` class wrap the `BERTEncoder` and adds a linear layer to predict class probabilities 

How are we doing it ?? 

- Extract the CLS token's embedding `(output[:, 0, :])` from the encoder's output, as BERT uses this for classification tasks.

- Applies dropout for regularization. 

2. Tokenizer integration : using `transformers.BertTokenizer` from hugging face, pre-trained on `bert-base-uncased`. 


## Experimenting with GPTDecoder

- Convert this whole into a hybrid model 

Maths here

Bert encoder : input token ID $x = [x_1, ... , x_n]$, n is the sequence length