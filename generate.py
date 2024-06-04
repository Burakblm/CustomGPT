import torch
from torch.nn import functional as F
import os
import time

from utils import get_tokenizer
from model import Transformer, ModelArgs
from lora import Lora

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = os.path.join(os.getcwd(), "model", "snapshot.pt")

model_args = ModelArgs()
model = Transformer(model_args)
model.to(device)

model.load_state_dict(torch.load(model_path, map_location=device))

tokenizer = get_tokenizer()

def generate_text(text: str, temperature: float = 1.0, top_k: int = 0, do_sample: bool = True, stop_token=[32000], max_token: int = 100):
    model.eval()
    idx = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=device)

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

            if idx_next in stop_token:
                break

            idx = torch.cat((idx, idx_next), dim=1)

        return tokenizer.decode(idx[0].tolist())

