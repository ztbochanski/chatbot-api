from input_parser import Encoder, Reader
from intent_predictor import Predictor
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, send

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
socketio = SocketIO(app, cors_allowed_origins="*")


@socketio.on('message')
def handle_message(data):
    print('received message: ' + data)

# @socketio.on('message')
# def handle_message(message):
#     send(message)


@app.route("/surgery-recovery/api/v1.0/assistant", methods=['POST'])
def predict_intent():

    user_data = request.get_json()
    user_input = user_data['input']

    encoder = Encoder()
    encoded_user_input = encoder.encode_input(user_input, stemmed_words)

    predictor = Predictor()
    predictor.predict(model, encoded_user_input)
    predictor.filter_predictions(PREDICTION_BUFFER)

    predictor.sort_predictions()

    prediction_hash = predictor.build_hash(labels)
    return jsonify(prediction_hash)


if __name__ == '__main__':
    socketIo.run(app, host='0.0.0.0', port=3000, use_reloader=True)
