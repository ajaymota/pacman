#!flask/bin/python

from flask import Flask, request, render_template, session, json, abort, jsonify
from flask_socketio import SocketIO, emit

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
    emit('server msg', {'data': 'Connected'})


@socketio.on('client msg', namespace='/test')
def test_message(message):
    print(message['data'])


@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')


@app.route('/')
def home():
    # serve index template
    return render_template('index.html')


if __name__ == '__main__':
    # run!
    socketio.run(app)
