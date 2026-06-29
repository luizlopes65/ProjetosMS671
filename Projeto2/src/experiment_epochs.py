import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from ingest_data import return_data
from model import build_model, compile_model
from preprocess_data import preprocess_human_data, preprocess_machine_data


SEED = 42
Tx = 30
Ty = 10
n_a = 32
n_s = 64
BATCH_SIZE = 64
EPOCH_CHECKPOINTS = [1, 5, 15, 25, 35, 50]


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def exact_match_percent(model, X_test_one_hot, y_test):
    m_test = X_test_one_hot.shape[0]
    s0_test = np.zeros((m_test, n_s), dtype=np.float32)
    c0_test = np.zeros((m_test, n_s), dtype=np.float32)

    predictions = model.predict(
        [X_test_one_hot, s0_test, c0_test],
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    predicted = np.stack(
        [np.argmax(timestep_prediction, axis=1) for timestep_prediction in predictions],
        axis=1,
    )

    return 100 * np.mean(np.all(predicted == y_test, axis=1))


def main():
    fix_seed(SEED)

    (X_train, y_train, X_test, y_test), human_vocab, mac_vocab = return_data(
        "data",
        random_state=SEED,
    )

    X_train = preprocess_human_data(X_train, human_vocab, Tx)
    X_test = preprocess_human_data(X_test, human_vocab, Tx)
    y_train = preprocess_machine_data(y_train, mac_vocab, Ty)
    y_test = preprocess_machine_data(y_test, mac_vocab, Ty)

    X_train_one_hot = to_categorical(X_train, num_classes=len(human_vocab))
    X_test_one_hot = to_categorical(X_test, num_classes=len(human_vocab))
    Y_train_one_hot = to_categorical(y_train, num_classes=len(mac_vocab))
    Y_test_one_hot = to_categorical(y_test, num_classes=len(mac_vocab))

    Y_train_outputs = [Y_train_one_hot[:, timestep, :] for timestep in range(Ty)]
    Y_test_outputs = [Y_test_one_hot[:, timestep, :] for timestep in range(Ty)]

    m_train = X_train_one_hot.shape[0]
    m_test = X_test_one_hot.shape[0]
    s0_train = np.zeros((m_train, n_s), dtype=np.float32)
    c0_train = np.zeros((m_train, n_s), dtype=np.float32)
    s0_test = np.zeros((m_test, n_s), dtype=np.float32)
    c0_test = np.zeros((m_test, n_s), dtype=np.float32)

    model = build_model(
        Tx=Tx,
        Ty=Ty,
        human_vocab_size=len(human_vocab),
        machine_vocab_size=len(mac_vocab),
        n_a=n_a,
        n_s=n_s,
    )
    compile_model(model)

    print(f"Seed fixa: {SEED}")
    print(f"Treino: {m_train} exemplos | Teste: {m_test} exemplos")
    print("epocas,loss_teste,exact_match_teste_%")

    trained_epochs = 0
    for checkpoint in EPOCH_CHECKPOINTS:
        epochs_to_train = checkpoint - trained_epochs
        model.fit(
            x=[X_train_one_hot, s0_train, c0_train],
            y=Y_train_outputs,
            epochs=epochs_to_train,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            verbose=2,
        )
        trained_epochs = checkpoint

        results = model.evaluate(
            x=[X_test_one_hot, s0_test, c0_test],
            y=Y_test_outputs,
            batch_size=BATCH_SIZE,
            verbose=0,
        )
        exact_match = exact_match_percent(model, X_test_one_hot, y_test)
        print(f"{checkpoint},{results[0]:.6f},{exact_match:.2f}")


if __name__ == "__main__":
    main()
