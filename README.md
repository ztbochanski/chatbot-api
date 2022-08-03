# surgery-recovery-assistant-api

```
surgery-recovery-assistant-api
│   README.md
│      
└───api
│   │   ...
│    
└───tests
│   │   ...
│    
└───training
│   │   ...
```

## Table of contents
- [surgery-recovery-assistant-api](#surgery-recovery-assistant-api)
  - [Table of contents](#table-of-contents)
  - [Tools](#tools)
  - [Training the Agent](#training-the-agent)
    - [Load Intent](#load-intent)
    - [Stem Words](#stem-words)
      - [Algorithm: Parse raw, domain specific intents structure](#algorithm-parse-raw-domain-specific-intents-structure)
  - [Generate Training Data](#generate-training-data)
    - [**Encode word patterns**](#encode-word-patterns)
      - [Algorithm:](#algorithm)
  - [Transform Data](#transform-data)
    - [**Separate labels from features**](#separate-labels-from-features)
  - [Create Model](#create-model)
      - [Compile](#compile)
      - [Train](#train)
      - [Plot performance](#plot-performance)
  - [Process Input](#process-input)
  - [Save Model](#save-model)
   
## Tools
- `json`
- `tensorflow as tf`
- `nltk.stem.lancaster -> LancasterStemmer`
- `numpy`
- `pandas`
- `sklearn.model_selection -> train_test_split`
- `matplotlib.pyplot`
- `pickle`
- `nltk`
- `nltk.download('punkt')` <-Library to tokenize words

## Training the Agent

One of the key features of an chatbot/assistant type program is its ability to recognize the intent that a user has when interacting with it. We can manually define "intent" and then build a corpus to be used in training an agent that can recognize the domain specific intent of a user.

1. Load intent
2. Stem words from intent
3. Determine predictors (labels) and container of stemmed words
4. Generate Training data (matrix of 0s and 1s representing word patterns)
5. Compile model
6. Fit model
7. Save model

### Load Intent

```py
with open('intents.json') as data:
    intents = json.load(data)
```

### Stem Words

Stem words using the stemming library to create word roots
- `tokens = []`
- `labels = []`
- `entities = []`

**Entities**
>Entities are building blocks of intent-based systems that allow you to extract and categorize information from a string of text or phrase.

```py
stemmer = LancasterStemmer()
tokens = []
labels = []
entities = []
ignore_tokens = []
```

#### Algorithm: Parse raw, domain specific intents structure

1. Tokenize phrase

`token = nltk.word_tokenize(pattern)`

2. Add tokenized word to list

`tokens.extend(token)`

3. Create the corpus by filling with entities

`entities.append((token, intent['tag']))`

4. Each tag becomes a class label

`if intent['tag'] not in labels: labels.append(intent['tag'])`

## Generate Training Data

>Take all the words and turn them into mini containers of 0s and 1s. The 1s are for matches of where a root word in an entity matches a word in the list of total words; so each entity has its own container of matches. 

### **Encode word patterns**

To create the training data out of all the entities in the corpus (structure of categorized words to a label):

1. Make a separate "word container" for each entity.
2. For the pattern words in each entity tokenize and stem them. (create word roots) to match the word roots in words list.
3. For all words, each one in the entire intent (tokenized and stemmed from above) create a fill this separate "word container" with 0s except for where each one of these words matches a word in the pattern for the specific entity. In that case put a 1 in the array.

One iteration through this algorithm creates a container full of 0s with 1s marked where there is a match between a word in the tagged entity (a pattern we know about) and the array of all words. This is how we generate data points. These containers all container "highlighted" words (1s) within a big container of words (0s). These highlighted words contain patterns from which we can then tune our machine learning model to recognize and pair with a label. In this sense we are just clustering words by similarity and classifying the clusters which represent an intent.

**pattern words:**
1s = stemmed word occurrence

**entities:**
1s = occurrence of tag

**data** - stemmed words array, label array

**tags** - classifier label

#### Algorithm:
- Get the array of tokenized words from the specific pattern in entities

- Stem these words to identify related words
    
- Create an array of 0s for the patterns, add a 1 word match found in current pattern
    
- Place a 1 for the current tag

## Transform Data

- Create dataframe from features and labels and combine.

-  Examine dataframe for NANs and shape.

```py
df_features = pd.DataFrame(features, columns=[stemmed_words])
df_labels = pd.DataFrame(classes, columns=[labels])
df = df_features.merge(df_labels, how="inner",
                       left_index=True, right_index=True)
print(df.shape)
print(df.isna().sum())
```

### **Separate labels from features**

Further split X (features) and y (class labels) into training and testing data using random sample selection.

Separate X (features) data from y (target) values:
- X: arrays of stemmed word occurrences (1s) in each list of 0s
- y: arrays of all labels, 1 indicating the type of label

Split the resulting data into train and test data sets:
- `train_test_split(X, y, test_size = 0.2, random_state = 0)`

**Separate**
```py
slice_size = len(labels)
X = df.iloc[:, :-slice_size]
y = df.iloc[:, -slice_size:]
```
**Split**
```py
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
```

**Confirm**
```py
print(X.shape)
print(y.shape)
```

## Create Model

Experimentation with different layer configurations and different numbers of internal neurons alters results. 

One important aspect of this recurrent neural network is that it contains one dense input layer that is the size (shape) of its features. These are the stemmed word occurances. 

The output is predicting the intent (one of the 9 labels), using an activation function that is friendly towards classification like `softmax`.

```py
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu',
                          input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(.4),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(.3),
    tf.keras.layers.Dense(y_train.shape[1], activation='softmax'),
])
```

#### Compile
```py
model.compile()
```

#### Train
```py
history = model.fit()
```

#### Plot performance

- Loss
- Accuracy

## Process Input

Example:

```py
test_user_input = "Where can I get my medications?"  # pharmacy search intended

encoded_user_input = encode_input(test_user_input, stemmed_words)
print("Encoded input:", encoded_user_input)
print("Intent labels:", labels)
```

**Prediction using the model:**

- predict intent (highest probability)


```py
df_input = pd.DataFrame([encoded_user_input],
                        dtype=float, index=['user_input'])
predictions = model.predict(df_input)
preds = predictions.flatten().tolist()

for label in range(len(labels)):
    print("Predicted intent: " +
          "{:.06%}".format(preds[label]) + " \"" + labels[label] + "\"")
```

## Save Model

- use a storage format to save the pre-trained model

```py
pickle.dump(model, open("surgery-recovery-assistant-model.pkl", "wb"))

pickle.dump({'stemmed_words': stemmed_words, 'labels': labels, 'X_train': X_train,
            'y_train': y_train}, open('surgery-recovery-assistant-data.pkl', 'wb'))
```