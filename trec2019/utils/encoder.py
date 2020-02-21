import torch
from transformers import BertModel
from transformers import BertTokenizer
from abc import ABCMeta


class Encoder(metaclass=ABCMeta):
    def __init__(self):
        pass

    def encode(self, text):
        pass


class BertEncoder(Encoder):
    MAX_LENGTH = 512

    def __init__(self, weights="bert-base-uncased"):
        self.model = BertModel.from_pretrained(weights)
        self.tokenizer = BertTokenizer.from_pretrained(weights)

    def encode(self, text):
        input_ids = torch.tensor(
            [
                self.tokenizer.encode(
                    text, add_special_tokens=True, max_length=BertEncoder.MAX_LENGTH,
                )
            ]
        )
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]
        return last_hidden_states
