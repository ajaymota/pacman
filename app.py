#!flask/bin/python

from flask import Flask, request, render_template, session, json, abort, jsonify
from flask_socketio import SocketIO, emit
from itertools import chain

app = Flask(__name__)
socketio = SocketIO(app)


@app.route('/hello')
def output():
    # serve index template
    return "Hello World!"


@app.route('/test')
def test():
    # serve index template
    text = request.args.get('text')
    return text


@socketio.on('connect', namespace='/test')
def test_connect():
    emit('connect msg', {'data': 'Connected'})


@socketio.on('client msg', namespace='/test')
def test_message(message):
    features = extract_features(message['data'])
    send_dir()


@socketio.on('my event', namespace='/test')
def send_dir():
    emit('server msg', {'data': 0.67})


@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')


@app.route('/')
def home():
    # serve index template
    return render_template('index.html')


def extract_features(data):
    """
    data[0] = Ghost Distance; len = 4 (each index = ghost)
    data[1] = Ghost Directions; len = 4 (each index = ghost)
    data[2] = Ghost status; len = 4 (each index = ghost)
    data[3] = Nearest Biscuit; len = 4 (each index = direction {UP, RIGHT, DOWN, LEFT})
    data[4] = Nearest Wall; len = 4 (each index = direction {UP, RIGHT, DOWN, LEFT})
    data[5] = User position; len = 2 (x, y)
    :param data:
    :return:
    """
    data = list(chain.from_iterable(data))
    return data


if __name__ == '__main__':
    # run!
    socketio.run(app)
