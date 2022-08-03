import json
import tensorflow as tf
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import nltk

# ######################################################################################################################
# LOAD DATA
nltk.download('punkt')
with open('training_data/intents.json') as data:
    intents = json.load(data)

# ######################################################################################################################
# STEM WORDS

stemmer = LancasterStemmer()
tokens = []
labels = []
entities = []
ignore_tokens = ['?']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        token = nltk.word_tokenize(pattern)
        tokens.extend(token)
        entities.append((token, intent['tag']))
        if intent['tag'] not in labels:
            labels.append(intent['tag'])

stemmed_words = [stemmer.stem(token.lower())
                 for token in tokens if token not in ignore_tokens]
stemmed_words = sorted(list(set(stemmed_words)))

labels = sorted(list(set(labels)))

print(len(entities), 'entities')
print(len(labels), 'labels', labels)
print(len(stemmed_words), 'unique stemmed_words', stemmed_words)

# ######################################################################################################################
# ENCODE STEMMED WORDS

features = []
classes = []
tags = [0] * len(labels)

for entity in entities:
    word_container = []
    pattern_words = entity[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    for word in stemmed_words:
        word_container.append(
            1) if word in pattern_words else word_container.append(0)

    tag_row = list(tags)
    tag_row[labels.index(entity[1])] = 1

    features.append(word_container)
    classes.append(tag_row)

# ######################################################################################################################
# CREATE DATAFRAME

df_features = pd.DataFrame(features, columns=[stemmed_words])
df_labels = pd.DataFrame(classes, columns=[labels])
df = df_features.merge(df_labels, how='inner',
                       left_index=True, right_index=True)
print(df.shape)
print(df.isna().sum())

slice_size = len(labels)
X = df.iloc[:, :-slice_size]
y = df.iloc[:, -slice_size:]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

print(X.shape)
print(y.shape)

# ######################################################################################################################
# CREATE MODEL

tf.random.set_seed(42)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu',
                          input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(.4),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(.3),
    tf.keras.layers.Dense(y_train.shape[1], activation='softmax'),
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True),
              loss='categorical_crossentropy', metrics=['acc'])

# ######################################################################################################################
# TRAIN
history = model.fit(np.array(X_train), np.array(y_train),
                    batch_size=5, epochs=180, verbose=1)

plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper right')
plt.show()

plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='lower right')
plt.show()

# ######################################################################################################################
# SAVE MODEL

pickle.dump(model, open(
    'training/training_data/surgery-recovery-assistant-model.pkl', 'wb'))

pickle.dump({'stemmed_words': stemmed_words, 'labels': labels, 'X_train': X_train,
            'y_train': y_train}, open('training/training_data/surgery-recovery-assistant-data.pkl', 'wb'))
