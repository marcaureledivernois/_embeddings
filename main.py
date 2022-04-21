import nltk
nltk.download("punkt");
import numpy as np
import re
import urllib
import gensim
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

SEED = 1234
np.random.seed(SEED)

# Split text into sentences
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
book = urllib.request.urlopen(url="https://raw.githubusercontent.com/GokuMohandas/MadeWithML/main/datasets/harrypotter.txt")
sentences = tokenizer.tokenize(str(book.read()))
print(len(sentences), "sentences")

def preprocess(text):
    """Conditional preprocessing on our text."""
    # Lower
    text = text.lower()

    # Spacing and filters
    text = re.sub(r"([-;;.,!?<=>])", r" \1 ", text)
    text = re.sub("[^A-Za-z0-9]+", " ", text) # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()

    # Separate into word tokens
    text = text.split(" ")

    return text


# Preprocess sentences
print (sentences[11])
sentences = [preprocess(sentence) for sentence in sentences]
print (sentences[11])

