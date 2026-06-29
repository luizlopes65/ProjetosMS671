import numpy as np
from tensorflow.keras.utils import to_categorical

from ingest_data import return_data
from model import build_model, compile_model
from preprocess_data import (
    preprocess_human_data,
    preprocess_machine_data,
)
from plot_attention_map import 

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

# Descobre o número total de exemplos de treino e teste
m_treino = X_train_one_hot.shape[0]
m_teste = X_teste.shape[0]

# Estados iniciais zerados para o treino
s0_treino = np.zeros((m_treino, n_s), dtype=np.float32)
c0_treino = np.zeros((m_treino, n_s), dtype=np.float32)

# Estados iniciais zerados para o teste
s0_teste = np.zeros((m_teste, n_s), dtype=np.float32)
c0_teste = np.zeros((m_teste, n_s), dtype=np.float32)

EPOCHS = 30       # Quantas vezes o modelo vai ver todo o dataset
BATCH_SIZE = 64   # Quantos exemplos ele processa por vez antes de atualizar os pesos

print("\n--- Iniciando o Treinamento ---")
historico = model.fit(
    x=[X_train_one_hot, s0_treino, c0_treino], # As 3 entradas exigidas
    y=Y_train_outputs,                         # A lista com as 10 saídas esperadas
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1                       # Separa 10% para validação durante o treino
)

# Prepara as saídas de teste no formato de lista (Ty elementos)
Y_test_outputs = [
    to_categorical(y_teste, num_classes=len(mac_vocab))[:, timestep, :]
    for timestep in range(Ty)
]

print("\n--- Avaliando o Modelo com Dados de Teste ---")
resultados = model.evaluate(
    x=[to_categorical(X_teste, num_classes=len(human_vocab)), s0_teste, c0_teste],
    y=Y_test_outputs,
    batch_size=BATCH_SIZE
)

def traduzir_data(data_humana):
    # 1. Coloca a string dentro de um array e pré-processa
    data_array = np.array([data_humana])
    data_proc = preprocess_human_data(data_array, human_vocab, Tx)
    
    # 2. Transforma em One-Hot
    data_one_hot = to_categorical(data_proc, num_classes=len(human_vocab))
    
    # 3. Cria os estados iniciais zerados (batch_size = 1 neste caso)
    s0_pred = np.zeros((1, n_s), dtype=np.float32)
    c0_pred = np.zeros((1, n_s), dtype=np.float32)
    
    # 4. Faz a previsão
    previsoes = model.predict([data_one_hot, s0_pred, c0_pred])
    
    # 5. O modelo retorna uma lista de 10 characteres. 
    # Precisamos pegar o índice de maior probabilidade (argmax) para cada um deles.
    indices_saida = [np.argmax(caractere[0]) for caractere in previsoes]
    
    # 6. Inverte o dicionário mac_vocab para transformar índices de volta em letras
    inv_mac_vocab = {v: k for k, v in mac_vocab.items()}
    
    # 7. Junta os caracteres na string final
    data_formatada = "".join([inv_mac_vocab[idx] for idx in indices_saida])
    
    print(f"Entrada: {data_humana} -> Saída do Modelo: {data_formatada}")

# Exemplo de teste após o modelo estar treinado:
traduzir_data("3 May 1995")
traduzir_data("05/12/2021")
traduzir_data("12 May 1998")           # Formato limpo, por extenso
traduzir_data("Jan 25, 2015")          # Mês primeiro, com vírgula e abreviado
traduzir_data("20 AUG 1985")           # Mês abreviado em maiúsculas
traduzir_data("Sunday, 11 March 2007") # Incluindo o dia da semana (ruído extra)
traduzir_data("3 de Julho de 1974")    # Formato longo com conectivos "de"
traduzir_data("1 de fevereiro 2026")   # Mês por extenso em minúsculo
traduzir_data("30/11/1999")            # Barras (padrão BR)
traduzir_data("08.12.2004")            # Pontos
traduzir_data("15-04-2010")            # Hifens
traduzir_data("02 Sep 21")             # Ano com apenas 2 dígitos (2021)
