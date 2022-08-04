from flask_cors import CORS
from input_parser import Encoder, Reader
from intent_predictor import Predictor
from data_manager import Corpus
from flask import Flask, jsonify, request
from flask_cors import CORS

PREDICTION_BUFFER = .2
DATA_PATH = 'training/training_data/surgery-recovery-assistant-data.pkl'
MODEL_PATH = 'training/training_data/surgery-recovery-assistant-model.pkl'

reader = Reader()
reader.read_data(DATA_PATH)
reader.read_model(MODEL_PATH)

model = reader.get_model()
stemmed_words = reader.get_stemmed_words()
labels = reader.get_labels()

app = Flask(__name__)
CORS(app)


@app.route('/surgery-recovery/api/v1.0/random_reply', methods=['POST'])
def random_reply():
    data = request.get_json()
    tag = data['tag']
    corpus = Corpus(directory='training/training_data/intents.json')
    return jsonify(corpus.random_response(tag))


@app.route('/surgery-recovery/api/v1.0/predict_intent', methods=['POST'])
def predict_intent():
    data = request.get_json()
    user_input = data['input']

    encoder = Encoder()
    encoded_user_input = encoder.encode_input(user_input, stemmed_words)

    predictor = Predictor()
    predictor.predict(model, encoded_user_input)
    predictor.filter_predictions(PREDICTION_BUFFER)

    predictor.sort_predictions()

    prediction_hash = predictor.build_hash(labels)
    return jsonify(prediction_hash)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
