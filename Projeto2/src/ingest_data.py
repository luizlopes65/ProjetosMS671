import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import json



def partition_data(data_csv, test_size=0.1):
    X_train, X_test, y_train, y_test = train_test_split(
        data_csv[:, 0],
        data_csv[:, 1],
        test_size=test_size,
    )
    return X_train, y_train, X_test, y_test

def return_data(path):
    data_path = os.path.join(path, "dataset.csv")
    data = pd.read_csv(data_path)
    human_vocab_path = os.path.join(path, "vocabularies/human_vocab.json")
    machine_vocab_path = os.path.join(path, "vocabularies/machine_vocab.json")

    with open(human_vocab_path, "r", encoding="utf-8") as f:
        human_vocab = json.load(f)
    with open(machine_vocab_path, "r", encoding="utf-8") as f:
        machine_vocab = json.load(f)
    data = partition_data(data.to_numpy())

    return data, human_vocab, machine_vocab
