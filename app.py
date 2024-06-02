from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")

def background_thread():
    count = 0
    while True:
        time.sleep(2)
        count += 1
        socketio.emit('message', {'data': f'Server generated message {count}'}, namespace='/')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('message', {'data': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('chat_message')
def handle_chat_message(msg):
    print(f'Received message: {msg}')
    emit('message', msg, broadcast=True) 

if __name__ == '__main__':
    thread = threading.Thread(target=background_thread)
    thread.daemon = True
    thread.start()
    socketio.run(app, debug=True)