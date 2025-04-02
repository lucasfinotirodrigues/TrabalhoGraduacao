import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Carregar os vetores extraídos
with open("train_features.pkl", "rb") as f:
    X_train, y_train = pickle.load(f)

with open("test_features.pkl", "rb") as f:
    X_test, y_test = pickle.load(f)

# Converter para numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)

# Codificar as classes
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Criar um DataFrame para armazenar os resultados
results_df = pd.DataFrame()

for i in range(100):
    print(f"Treinamento {i+1}/100")
    
    # Criar o modelo
    model = Sequential([
        Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(y_train.shape[1], activation='softmax')
    ])

    # Compilar o modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Treinar o modelo
    model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    # Avaliação do modelo
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    # Obter o relatório de classificação
    report = classification_report(y_test_classes, y_pred_classes, target_names=label_encoder.classes_, output_dict=True)
    
    # Converter os dados para um DataFrame temporário
    temp_df = pd.DataFrame(report).transpose()
    temp_df.insert(0, "Run", i+1)  # Adicionar uma coluna indicando a iteração
    
    # Concatenar os resultados ao DataFrame principal
    results_df = pd.concat([results_df, temp_df])

# Salvar os resultados em um CSV
results_df.to_csv("training_results.csv", index=True)
print("Resultados salvos em 'training_results.csv'!")
