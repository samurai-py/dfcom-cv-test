# dfcom-cv-test

## Resumo do Projeto
O projeto **dfcom-cv-test** é uma aplicação de visão computacional que utiliza aprendizado profundo para classificação de imagens. Ele inclui scripts para pré-processamento de imagens, treinamento de modelos de aprendizado profundo, e uma interface para fazer previsões em novas imagens usando modelos treinados. Além disso, na pasta `scripts`, é possível acessar o notebook original do projeto.

## Estrutura de Diretórios
```plaintext
dfcom_cv
├── app
│   ├── app.py
│   └── retrainer.py
├── __init__.py
├── models
│   ├── cnn
│   │   └── modelo_cnn.h5
│   └── transfer
│       └── modelo_transfer.h5
├── predict
│   └── predict.py
├── preprocessing
│   ├── image_transforming.py
│   └── loader.py
├── train
│   ├── base_trainer.py
│   ├── cnn_trainer.py
│   └── transfer_trainer.py
├── LICENSE
├── pyproject.toml
├── README.md
├── scripts
│   └── models.ipynb
└── setup.py
```

## Instalação
Para instalar o projeto, siga os passos abaixo:

1. Clone o repositório:
    ```bash
    git clone https://github.com/samurai-py/dfcom-cv-test.git
    ```

2. Navegue até o diretório do projeto:
    ```bash
    cd dfcom-cv-test
    ```

3. Crie um ambiente virtual:
    ```bash
    python3 -m venv venv
    ```

4. Ative o ambiente virtual:
    - No Linux/Mac:
        ```bash
        source venv/bin/activate
        ```
    - No Windows:
        ```bash
        .\venv\Scripts\activate
        ```

5. Instale as dependências:
    ```bash
    pip install -e .
    ```

## Execução
Para executar o projeto, utilize o comando abaixo:

```bash
start_app
```

## Ferramentas Utilizadas
- **Python**: Linguagem de programação principal.
- **Pandas**: Biblioteca para manipulação e análise de dados.
- **Tensorflow**: Biblioteca para machine learning.
- **NumPy**: Biblioteca para computação numérica.
- **Streamlit**: Biblioteca para criação de aplicações web em python.
- **Ruff**: Ferramenta para formatação e padronização de código

---

