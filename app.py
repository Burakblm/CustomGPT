import torch
from torch.nn import functional as F
import os
from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import time

from utils import get_tokenizer
from model import Transformer, ModelArgs
from lora import Lora

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('message', {'data': 'Sunucuya Bağlandı'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('chat_message')
def handle_chat_message(data):
    msg = data['text']
    temperature = data.get('temperature', 1.0)
    top_k = data.get('top_k', 0)
    do_sample = data.get('do_sample', True)
    max_new_token = data.get('token_selection', 200)
    print(f'Alınan mesaj: {msg}, Temperature: {temperature}, Top_k: {top_k}, Do_sample: {do_sample} max_token: {max_new_token}')
    generate_text_stream(msg, temperature=temperature, top_k=top_k, do_sample=do_sample, max_token=max_new_token)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = os.path.join(os.getcwd(), "model", "snapshot.pt")

model_args = ModelArgs()
model = Transformer(model_args)
model.to(device)

model.load_state_dict(torch.load(model_path, map_location=device))

tokenizer = get_tokenizer()

def generate_text_stream(text: str, temperature: float = 1.0, top_k: int = 0, do_sample: bool = True, stop_token=[32000], max_token: int = 100):
    model.eval()
    idx = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=device)
    text_arr = tokenizer.encode(text)
    count_first = text.count(' ')
    count_second = text.count(' ')


    with torch.no_grad():
        for _ in range(max_token):
            logits, _ = model(idx)
            logits = logits[:, -1, :] / temperature

            if top_k is not None and top_k != 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-Inf")

            props = F.softmax(logits, dim=-1)

            if do_sample:
                idx_next = torch.multinomial(props, num_samples=1)
            else:
                _, idx_next = torch.topk(props, k=1, dim=-1)

            text_arr.append(idx_next[0].tolist()[0])
            text = tokenizer.decode(text_arr)
            count_second = text.count(' ')
            if count_first != count_second:
                word = text.split(' ')[-2]
                socketio.emit('new_word', {'word': word})
                count_first = count_second

            if idx_next in stop_token:
                break

            idx = torch.cat((idx, idx_next), dim=1)
        print(tokenizer.decode(idx[0].tolist()))

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5003, allow_unsafe_werkzeug=True)