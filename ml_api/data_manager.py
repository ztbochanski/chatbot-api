import json
import random


class Corpus:

    def __init__(self, directory=None):
        self.directory = directory
        with open(self.directory) as data:
            intents = json.load(data)
        self.intents = intents

    def tags(self):
        tags = []
        for entity in self.intents['intents']:
            tags.append(entity['tag'])
        return tags

    def responses(self, tag):
        for entity in self.intents['intents']:
            if entity['tag'] == tag:
                return entity['responses']

    def random_response(self, tag):
        responses = self.responses(tag)
        random_number = random.randint(0, len(responses)-1)
        response = responses[random_number]
        return response


if __name__ == '__main__':
    corpus = Corpus(directory='training/training_data/intents.json')
    print(corpus.random_response('pharmacy_search'))
