from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import nltk
import pickle


class Encoder:

    def __init__(self) -> None:
        nltk.download('punkt')
        self.stemmer = LancasterStemmer()

    def tokenize_stem_input(self, user_input):
        tokenized_input = nltk.word_tokenize(user_input)
        return [self.stemmer.stem(word.lower()) for word in tokenized_input]

    def encode_input(self, user_input, all_model_words):
        tokenized_input = self.tokenize_stem_input(user_input)

        word_matrix = [0]*len(all_model_words)
        for stemmed_word in tokenized_input:
            for i, word in enumerate(all_model_words):
                if word == stemmed_word:
                    word_matrix[i] = 1
                    print('User input found in corpus: %s' % word)

        return(np.array(word_matrix))


class Reader:

    def __init__(self, model=None, stemmed_words=None, labels=None):
        self.model = model
        self.stemmed_words = stemmed_words
        self.labels = labels

    def set_model(self, model):
        self.model = model

    def get_model(self):
        return self.model

    def set_stemmed_words(self, stemmed_words):
        self.stemmed_words = stemmed_words

    def get_stemmed_words(self):
        return self.stemmed_words

    def set_labels(self, labels):
        self.labels = labels

    def get_labels(self):
        return self.labels

    def read_data(self, path):
        with open(path, 'rb') as data_file:
            data = pickle.load(data_file)
            self.set_stemmed_words(data['stemmed_words'])
            self.set_labels(data['labels'])

    def read_model(self, path):
        with open(path, 'rb') as model_file:
            self.set_model(pickle.load(model_file))
