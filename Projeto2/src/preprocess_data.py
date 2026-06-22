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
