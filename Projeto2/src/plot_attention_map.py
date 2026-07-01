import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# Funções auxiliares para mapear texto para inteiro e vice-versa
def _string_to_int(string, length, vocabulary):
    string = string.lower()
    string = string.replace(',','')
    # Se o caractere não existir no vocabulário, usa um valor padrão ou ignora
    rep = [vocabulary.get(char, 0) for char in string]
    if len(rep) < length:
        rep += [vocabulary.get('<pad>', 0)] * (length - len(rep))
    elif len(rep) > length:
        rep = rep[:length]
    return rep

def _int_to_string(ints, inv_vocabulary):
    return [inv_vocabulary.get(idx, '<unk>') for idx in ints]

def plot_attention_map(modelx, input_vocabulary, output_vocabulary, text, n_s = 64, num = 7):
    """
    Plot the attention map.
    """
    attention_map = np.zeros((10, 30))
    Ty, Tx = attention_map.shape

    # 1. Obter as entradas originais do modelo
    X = modelx.inputs[0]
    s0 = modelx.inputs[1]
    c0 = modelx.inputs[2]
    s = s0
    c = c0

    a = modelx.layers[2](X)
    outputs = []

    # Reconstruindo o fluxo para capturar os pesos de atenção (alphas)
    for t in range(Ty):
        s_prev = s
        s_prev = modelx.layers[3](s_prev)
        concat = modelx.layers[4]([a, s_prev])
        e = modelx.layers[5](concat)
        energies = modelx.layers[6](e)
        alphas = modelx.layers[7](energies) # Camada Softmax da Atenção
        context = modelx.layers[8]([alphas, a])
        s, _, c = modelx.layers[10](context, initial_state = [s, c])
        
        # Guardamos 'alphas' (pesos de atenção) em vez de 'energies' para o mapa fazer sentido
        outputs.append(alphas)

    # CORREÇÃO DO ERRO: Usar a classe Model importada do Keras
    f = Model(inputs=[X, s0, c0], outputs=outputs)

    # 2. Preparação dos dados de entrada
    s0_val = np.zeros((1, n_s))
    c0_val = np.zeros((1, n_s))
    
    encoded = np.array(_string_to_int(text, Tx, input_vocabulary)).reshape((1, 30))
    encoded = np.array(list(map(lambda x: to_categorical(x, num_classes=len(input_vocabulary)), encoded)))

    # Fazer a predição dos pesos de atenção
    r = f([encoded, s0_val, c0_val])

    for t in range(Ty):
        for t_prime in range(Tx):
            # r[t] tem formato (batch_size, Tx, 1) -> tiramos o squeeze para mapear
            attention_map[t][t_prime] = np.squeeze(r[t])[t_prime]

    # Normalizar o mapa de atenção
    row_max = attention_map.max(axis=1)
    # Evita divisão por zero caso haja alguma linha zerada
    row_max[row_max == 0] = 1.0
    attention_map = attention_map / row_max[:, None]

    # Prever o texto final para colocar nos eixos do gráfico
    prediction = modelx.predict([encoded, s0_val, c0_val], verbose=0)
    
    predicted_text = []
    for i in range(len(prediction)):
        predicted_text.append(int(np.argmax(prediction[i], axis=1)[0]))

    inv_output_vocabulary = {v: k for k, v in output_vocabulary.items()}
    predicted_text = _int_to_string(predicted_text, inv_output_vocabulary)
    text_ = list(text)

    input_length = len(text)
    output_length = Ty

    # 3. Plotagem do Gráfico usando Matplotlib
    plt.clf()
    fig = plt.figure(figsize=(8, 8.5))
    ax = fig.add_subplot(1, 1, 1)

    i = ax.imshow(attention_map[:, :input_length], interpolation='nearest', cmap='Blues')

    cbaxes = fig.add_axes([0.2, 0.03, 0.6, 0.03])
    cbar = fig.colorbar(i, cax=cbaxes, orientation='horizontal')
    cbar.ax.set_xlabel('Alpha value (Probability output of the "softmax")', labelpad=2)

    ax.set_yticks(range(output_length))
    ax.set_yticklabels(predicted_text[:output_length])

    ax.set_xticks(range(input_length))
    ax.set_xticklabels(text_[:input_length], rotation=45)

    ax.set_xlabel('Input Sequence')
    ax.set_ylabel('Output Sequence')
    ax.grid(False) # Desliga linhas de grade sobrepostas ao mapa

    # Salva ou exibe o gráfico
    plt.savefig(filename, bbox_inches='tight')
    print("Mapa de atenção gerado com sucesso e salvo como '{filename}'!")
    plt.close()

    return attention_map
