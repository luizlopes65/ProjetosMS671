from ingest_data import return_data
from preprocess_data import preprocess_human_data

Tx = 30   # tamanho máximo da string de entrada
Ty = 10   # tamanho da saída YYYY-MM-DD

(X_treino, y_treino, X_teste, y_teste), human_vocab, mac_vocab = return_data("data")

X_treino_processado = preprocess_human_data(X_treino, human_vocab, Tx)

print(X_treino[0])
print(X_treino_processado[0])
print(y_treino[0])
