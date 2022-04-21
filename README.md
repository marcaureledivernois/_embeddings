# Embeddings

## Overview

Learn fixed length (=hyperparameter) representations of tokens
regardless of the number of tokens in the vocabulary. The intuition behind embeddings is that the definition of a token doesn't depend on the token itself but on its context.
Embeddings address the shortcomings 
of one-hot encoding (sparse arrays in the size of the vocabulary). Namely that one-hot encodings are linearly
dependent on the number of unique tokens in the vocabulary (issue if dealing with large corpus), and that tokens representation
do not keep any relationship with other tokens (=context). 

## Word2vec

Popular algorithm to learn embeddings. Uses two-layers neural network. Two variants: CBOW (continuous bag of words), skip-gram.

1. CBOW. training features : 2-tokens sized window around target word. training label : target word.
2. Skip-gram. training features : target word. training label : 2-tokens sized window around target word

Usually skip-gram performs better.

## Arithmetic property

        King - Man + Woman = Queen

Usually, we want to classify sentences, messages,... that consist of several words. In that case, we compute the **mean** of the word embeddings of a sentence.
This trick is a consequence of the arithmetic property of word embeddings.

## Credits

* [Made With ML](https://madewithml.com/#foundations)
