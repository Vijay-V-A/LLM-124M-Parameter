import tiktoken
import importlib.metadata
import torch
import numpy as np


class Tokenizer:
    def __init__(self, encoding_name="cl100k_base"):
        self.encoding_name = encoding_name
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str):
        """Encodes a string into token IDs."""
        encoded = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        return encoded_tensor

    def decode(self, tokens: list[int]):
        """Decodes token IDs back into a string."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.squeeze(0).tolist()

        return self.tokenizer.decode(tokens)

    def version(self):
        """Returns the installed tiktoken version."""
        return importlib.metadata.version("tiktoken")

    def list_encodings(self):
        """Lists all available encoding names."""
        return tiktoken.list_encoding_names()
