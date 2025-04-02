import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
import pickle

# Caminhos dos arquivos
dataset_folder = "./dataset/" 
excel_path = "Dataset.xlsx"  

# Carregar os dados do Excel
df = pd.read_excel(excel_path)

# Adicionar extensão se estiver faltando
df["Nome_da_Imagem"] = df["Nome_da_Imagem"].apply(lambda x: x if x.endswith(".png") else x + ".png")

# Separar treino e teste
df_train, df_test = train_test_split(df, test_size=0.3, stratify=df["Classe"], random_state=42)

# Carregar o modelo ResNet50 sem a camada final
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Função para extrair características
def extract_features(img_path):
    if not os.path.exists(img_path):
        print(f"Erro: Arquivo não encontrado -> {img_path}")
        return None
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = resnet_model.predict(img_array)
    return features.flatten()

# Testar os caminhos das primeiras imagens
for img in df_train["Nome_da_Imagem"].head(5):
    img_path = os.path.join(dataset_folder, img)
    print(f"Verificando: {img_path} -> Existe? {os.path.exists(img_path)}")

# Processar imagens de treino
df_train["features"] = df_train["Nome_da_Imagem"].apply(lambda img: extract_features(os.path.join(dataset_folder, img)))

# Processar imagens de teste
df_test["features"] = df_test["Nome_da_Imagem"].apply(lambda img: extract_features(os.path.join(dataset_folder, img)))

# Remover imagens não encontradas
df_train = df_train.dropna(subset=["features"])
df_test = df_test.dropna(subset=["features"])

# Salvar os vetores extraídos
with open("train_features.pkl", "wb") as f:
    pickle.dump((df_train["features"].tolist(), df_train["Classe"].tolist()), f)

with open("test_features.pkl", "wb") as f:
    pickle.dump((df_test["features"].tolist(), df_test["Classe"].tolist()), f)

print("Extração concluída! Os arquivos train_features.pkl e test_features.pkl foram salvos.")
