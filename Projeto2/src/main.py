import numpy as np
from tensorflow.keras.utils import to_categorical

from ingest_data import return_data
from model import build_model, compile_model
from preprocess_data import (
    preprocess_human_data,
    preprocess_machine_data,
)


Tx = 30  # tamanho máximo da string de entrada
Ty = 10  # tamanho da saída YYYY-MM-DD
n_a = 32  # unidades da LSTM em cada direção do encoder
n_s = 64  # unidades da LSTM decoder

(X_treino, y_treino, X_teste, y_teste), human_vocab, mac_vocab = return_data("data")

X_treino = preprocess_human_data(X_treino, human_vocab, Tx)
X_teste = preprocess_human_data(X_teste, human_vocab, Tx)
y_treino = preprocess_machine_data(y_treino, mac_vocab, Ty)
y_teste = preprocess_machine_data(y_teste, mac_vocab, Ty)

X_train_one_hot = to_categorical(
    X_treino,
    num_classes=len(human_vocab),
)
Y_train_one_hot = to_categorical(
    y_treino,
    num_classes=len(mac_vocab),
)
# O modelo retorna uma saída separada para cada uma das Ty posições.
Y_train_outputs = [
    Y_train_one_hot[:, timestep, :]
    for timestep in range(Ty)
]

model = build_model(
    Tx=Tx,
    Ty=Ty,
    human_vocab_size=len(human_vocab),
    machine_vocab_size=len(mac_vocab),
    n_a=n_a,
    n_s=n_s,
)
compile_model(model)

# Smoke test: somente uma passagem para frente, sem treinamento.
batch_size = 2
s0 = np.zeros((batch_size, n_s), dtype=np.float32)
c0 = np.zeros((batch_size, n_s), dtype=np.float32)
predictions = model(
    [X_train_one_hot[:batch_size], s0, c0],
    training=False,
)

print("Entrada one-hot:", X_train_one_hot.shape)
print("Alvo one-hot:", Y_train_one_hot.shape)
print("Shape de cada alvo:", Y_train_outputs[0].shape)
print("Número de saídas:", len(predictions))
print("Shape de cada saída:", predictions[0].shape)
print(
    "Soma das probabilidades da primeira saída:",
    float(np.sum(predictions[0][0])),
)
