from ingest_data import return_data
from preprocess_data import (
    preprocess_human_data,
    preprocess_machine_data,
)
from tensorflow.keras.utils import to_categorical


Tx = 30   # tamanho máximo da string de entrada
Ty = 10   # tamanho da saída YYYY-MM-DD

(X_treino, y_treino, X_teste, y_teste), human_vocab, mac_vocab = return_data("data")

X_treino, X_teste = preprocess_human_data(X_treino, human_vocab, Tx), preprocess_human_data(X_teste, human_vocab, Tx)
y_treino, y_teste = preprocess_machine_data(y_treino, mac_vocab, Ty), preprocess_machine_data(y_teste, mac_vocab, Ty)

X_one_hot_train = to_categorical(
    X_treino,
    num_classes=len(human_vocab),
)
Y_one_hot_train = to_categorical(
    y_treino,
    num_classes=len(mac_vocab),
)

print(X_one_hot_train.shape)