# Word embeddings

## Overview

Learn fixed length (=hyperparameter) representations of tokens
regardless of the number of tokens in the vocabulary. The intuition behind embeddings is that the definition of a token doesn't depend on the token itself but on its context.
Embeddings address the shortcomings 
of one-hot encoding (sparse arrays in the size of the vocabulary). Namely that one-hot encodings are linearly
dependent on the number of unique tokens in the vocabulary (issue if dealing with large corpus), and that tokens representation
do not keep any relationship with other tokens (=context). 

## Popular approaches

1. Word2Vec takes texts as training data for a neural network. The resulting embedding captures whether words appear in similar contexts.
2. GloVe focuses on words co-occurrences over the whole corpus. Its embeddings relate to the probabilities that two words appear together.
3. FastText improves on Word2Vec by taking word parts into account, too. This trick enables training of embeddings on smaller datasets and generalization to unknown words.

## Word2vec

Popular algorithm to learn embeddings. Uses two-layers neural network. Two variants: CBOW (continuous bag of words), skip-gram.

1. CBOW. training features : 2-tokens sized window around target word. training label : target word.
2. Skip-gram. training features : target word. training label : 2-tokens sized window around target word

Usually skip-gram performs better.

## Pretrained embeddings

Pretrained embeddings may be use either as initializing embeddings (and then can be fine-tuned)
or as final embeddings. Text files are available at [GloVe](https://nlp.stanford.edu/pubs/glove.pdf) for instance.

## Arithmetic property

        King - Man + Woman = Queen

Usually, we want to classify sentences, messages,... that consist of several words. In that case, we compute the **mean** of the word embeddings of a sentence.
This trick is a consequence of the arithmetic property of word embeddings.

## Credits

* [Made With ML](https://madewithml.com/#foundations)
