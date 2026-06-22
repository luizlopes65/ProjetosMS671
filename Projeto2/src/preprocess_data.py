import numpy as np


def preprocess_human_data(human_data, human_vocab, Tx):
    pad_id = human_vocab["<pad>"]
    unk_id = human_vocab["<unk>"]

    encoded = np.full((len(human_data), Tx), pad_id, dtype=np.int64)

    for i, text in enumerate(human_data):
        text = str(text).lower().replace(",", "")[:Tx]

        for j, char in enumerate(text):
            encoded[i, j] = human_vocab.get(char, unk_id)

    return encoded


def preprocess_machine_data(machine_data, machine_vocab, Ty):
    encoded = np.zeros((len(machine_data), Ty), dtype=np.int64)

    for i, text in enumerate(machine_data):
        text = str(text).strip()

        if len(text) != Ty or any(char not in machine_vocab for char in text):
            raise ValueError(
                f"Invalid machine date at index {i}: {text!r}. "
                f"Expected {Ty} characters using only {set(machine_vocab)}."
            )

        for j, char in enumerate(text):
            encoded[i, j] = machine_vocab[char]

    return encoded
