#!flask/bin/python

from flask import Flask, request, render_template
from flask_socketio import SocketIO, emit
from itertools import chain
import main
import pickle
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app, logging=False)

global generation
generation = 1


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
    games = message['data'][0]
    features = extract_features(message['data'][1])
    direction = main.interpret_output(main.use_model(features, (games+1)))
    send_dir(direction)


@socketio.on('games msg', namespace='/test')
def controller(message):
    games = message['data'][0]
    score = message['data'][1]
    global generation

    # print([games, score, generation])
    main.save_score((games-1), score)

    if games == 20:
        generation += 1
        with open("generation.db", "wb") as f:
            pickle.dump(generation, f)

        parents = main.get_fittest_parents()
        child = main.cross_over(parents[0], parents[1])
        main.rebreed_ten_models(child)


@socketio.on('my event', namespace='/test')
def send_dir(direction):
    emit('server msg', {'data': direction})


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
    data = np.array(data).T
    data = list(chain.from_iterable(data))
    return data


if __name__ == '__main__':
    # run!
    socketio.run(app)
