from tensorflow.keras import Model
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Concatenate,
    Dense,
    Dot,
    Input,
    RepeatVector,
    Softmax,
)
from tensorflow.keras.optimizers import Adam

Tx = 30


repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation="tanh", name="attention_dense")
densor2 = Dense(1, activation="relu", name="attention_energy")
activator = Softmax(axis=1, name="attention_weights")
dotor = Dot(axes=1)


def one_step_attention(a, s_prev):
    """Calcula o contexto de atenção para um passo do decoder."""
    s_prev_repeated = repeator(s_prev)

    concat = concatenator([a, s_prev_repeated])
    e = densor1(concat)
    energies = densor2(e)

    alphas = activator(energies)
    context = dotor([alphas, a])

    return context


def build_model(Tx, Ty, human_vocab_size, machine_vocab_size, n_a, n_s):
    """
    Constrói o modelo encoder-decoder com atenção.

    Entradas:
        X: datas em one-hot, shape (batch, Tx, human_vocab_size)
        s0: estado escondido inicial do decoder, shape (batch, n_s)
        c0: estado de célula inicial do decoder, shape (batch, n_s)

    Saída:
        Lista com Ty tensores de shape (batch, machine_vocab_size).
    """
    if Tx != repeator.n:
        raise ValueError(
            f"one_step_attention foi configurada para Tx={repeator.n}, "
            f"mas build_model recebeu Tx={Tx}."
        )

    X = Input(shape=(Tx, human_vocab_size), name="X")
    s0 = Input(shape=(n_s,), name="s0")
    c0 = Input(shape=(n_s,), name="c0")

    encoder = Bidirectional(
        LSTM(n_a, return_sequences=True),
        name="bidirectional_encoder",
    )
    decoder = LSTM(
        n_s,
        return_state=True,
        name="post_attention_lstm",
    )
    output_layer = Dense(
        machine_vocab_size,
        activation="softmax",
        name="output",
    )

    # a contém uma representação de cada posição da data de entrada.
    a = encoder(X)
    s = s0
    c = c0
    outputs = []

    # Um passo do decoder para cada caractere de YYYY-MM-DD.
    for _ in range(Ty):
        context = one_step_attention(a, s)
        s, _, c = decoder(context, initial_state=[s, c])
        outputs.append(output_layer(s))

    return Model(inputs=[X, s0, c0], outputs=outputs, name="date_translator")


def compile_model(model):
    """Configura o modelo para um futuro treinamento."""
    optimizer = Adam(learning_rate=0.005)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"] * len(model.outputs),
    )
    return model
