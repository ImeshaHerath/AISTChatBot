# Import Libraries
import nltk
import numpy as np
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer


stemmer = PorterStemmer()


# Splitting the string into meaningful units(words, numbers, ?,!)
def tokenize(sentence):
    return nltk.word_tokenize(sentence)


# Generate the root form of the word
def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):

    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    # array with all 0's [0,0,0,0,0,0,0] size = length of the all words
    bag = np.zeros(len(all_words), dtype=np.float32)
    # loop over our all words
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag  # [1,0,0,1]