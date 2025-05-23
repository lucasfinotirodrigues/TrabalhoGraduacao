# Trabalho de Graduação – Classificação de Amendoins

Este projeto é parte do Trabalho de Graduação de Lucas Finoti Rodrigues e tem como objetivo desenvolver um sistema de classificação de amendoins utilizando técnicas de aprendizado de máquina. O sistema realiza a classificação com base em imagens e características extraídas dos grãos.

## 📁 Estrutura do Projeto

- `app.py`: Script principal para execução da aplicação.
- `train_model.py` / `train_model_for.py`: Scripts de treinamento dos modelos de classificação.
- `test.py`: Script para testes do modelo treinado.
- `dataset/`: Conjunto de dados utilizado para treinamento e validação.
- `escala diagramática/`: Imagens de referência para classificação.
- `validacao/`: Dados e scripts relacionados à validação do modelo.
- `amendoim_model.h5`: Modelo treinado salvo em formato HDF5.
- `train_features.pkl` / `test_features.pkl`: Arquivos de features extraídas para treinamento e teste.
- `training_results.csv`: Resultados do treinamento, incluindo métricas de desempenho.
- `metodo-elbow.png`: Gráfico utilizado para determinar o número ideal de clusters no método Elbow.

## ⚙️ Tecnologias Utilizadas

- Python 3.x
- Bibliotecas:
  - TensorFlow / Keras
  - scikit-learn
  - NumPy
  - Pandas
  - Matplotlib
  - OpenCV

## 🚀 Como Executar

1. Clone o repositório:

   ```bash
   git clone https://github.com/lucasfinotirodrigues/TrabalhoGraduacao.git
   cd TrabalhoGraduacao

2. Instale as dependências
    ```bash
    pip install -r requirements.txt

3. Execute o script principal
    ```bash
    python app.py

