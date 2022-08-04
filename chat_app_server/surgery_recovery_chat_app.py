
from flask import Flask, render_template
from flask_socketio import SocketIO
import requests

DOMAIN = 'localhost'
PORT = '5000'

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')


@socketio.on('client event')
def handle_client_event(json, methods=['GET', 'POST']):
    print('received client event: ' + str(json))

    if not json['context']:

        assistant_http_response = requests.post('http://' + DOMAIN + ':' + PORT + '/surgery-recovery/api/v1.0/predict_intent', json={
            'input': json['message']
        })
        predicted_intent = assistant_http_response.json()[0]['intent']

        random_reply_http_response = requests.post('http://' + DOMAIN + ':' + PORT + '/surgery-recovery/api/v1.0/random_reply', json={
            'tag': predicted_intent
        })
        random_reply_message = random_reply_http_response.json()

        socketio.emit('server event', {
            'context': predicted_intent,
            'message': random_reply_message
        })
    else:
        random_reply_http_response = requests.post('http://' + DOMAIN + ':' + PORT + '/surgery-recovery/api/v1.0/random_reply', json={
            'tag': json['context']
        })
        print('context exists:', json['context'])
        random_reply_message = random_reply_http_response.json()

        socketio.emit('server event', {
            'context': json['context'],
            'message': random_reply_message
        })


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)
