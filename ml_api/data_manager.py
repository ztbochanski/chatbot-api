import json


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

    def patterns(self):
        patterns = []
        for entity in self.intents['intents']:
            patterns.append(entity['patterns'])
        return patterns

    def responses(self):
        responses = []
        for entity in self.intents['intents']:
            responses.append(entity['responses'])
        return responses

    def context(self):
        context = []
        for entity in self.intents['intents']:
            context.append(entity['context'])
        return context


if __name__ == '__main__':
    corpus = Corpus(directory='training/training_data/intents.json')
    print(corpus.context())
