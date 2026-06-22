# Projeto 2

Materiais iniciais do segundo projeto da disciplina MS671.

## Estrutura

```text
Projeto2/
├── data/
│   ├── dataset.pkl
│   └── vocabularies/
│       ├── human_vocab.pkl
│       ├── inv_machine_vocab.pkl
│       └── machine_vocab.pkl
├── models/
│   └── model.h5
└── src/
    ├── plot_attention_map.py
    └── string_to_int.py
```

## Conteúdo

- `data/`: conjunto de dados serializado.
- `data/vocabularies/`: vocabulários usados para converter entradas e saídas.
- `models/`: modelo Keras salvo em formato HDF5.
- `src/`: funções auxiliares para conversão de texto e visualização do mapa de atenção.

O arquivo `plot_attention_map.py` foi fornecido como função auxiliar e pressupõe
que NumPy, Matplotlib, Keras e as funções de conversão estejam disponíveis no
ambiente em que for importado.
