import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import pickle

# Carregar o modelo treinado
model = load_model("amendoim_model.h5")

# Carregar o codificador de rótulos
with open("train_features.pkl", "rb") as f:
    _, y_train = pickle.load(f)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(y_train)

# Carregar o extrator de características ResNet50
resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg")

def predict_image(img_path):
    # Carregar e pré-processar a imagem
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Extrair características com ResNet50
    features = resnet.predict(img_array)

    # Fazer a previsão com o modelo treinado
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction)

    # Obter o nome da classe prevista
    class_name = label_encoder.inverse_transform([predicted_class])[0]
    
    return class_name

# Testar com uma imagem
# caminho_imagem = 'validacao/img 1 - img 1.png'
# caminho_imagem = 'validacao/img 2 - img 5.png'
# caminho_imagem = 'validacao/img 3 - img 7.png'
# caminho_imagem = 'validacao/img 4 - img 15.png'
# caminho_imagem = 'validacao/img 5 - img-60.png'
# caminho_imagem = 'validacao/img 6 - img 77.png'
# caminho_imagem = 'validacao/img 7 - img 98.png'
# caminho_imagem = 'validacao/img 8 - img 183.png'
# caminho_imagem = 'validacao/img 9 - img 126.png'
# caminho_imagem = 'validacao/img 10 - img 148.png'
# caminho_imagem = 'validacao/img 11 - img 125.png'
# caminho_imagem = 'validacao/img 12 - img 159.png'
# caminho_imagem = 'validacao/img 13 - img 164.png'
# caminho_imagem = 'validacao/img 14 - img 182.png'
# caminho_imagem = 'validacao/img 15 - img 106.png'
# caminho_imagem = 'validacao/img 16 - img 211.png'
# caminho_imagem = 'validacao/img 17 - img 212.png'
# caminho_imagem = 'validacao/img 18 - img 214.png'
# caminho_imagem = 'validacao/img 19 - img 231.png'
# caminho_imagem = 'validacao/img 20 - img 233.png'
# caminho_imagem = 'validacao/img 21 - img 235.png'
# caminho_imagem = 'validacao/img 22 - img 252.png'
# caminho_imagem = 'validacao/img 23 - img 273.png'
# caminho_imagem = 'validacao/img 24 - img 278.png'
# caminho_imagem = 'validacao/img 25 - img 279.png'
# caminho_imagem = 'validacao/img 26 - img 301.png'
# caminho_imagem = 'validacao/img 27 - img 353.png'
# caminho_imagem = 'validacao/img 28 - img 355.png'

img_path = "./validacao/img 19 - img 231.png"
resultado = predict_image(img_path)
print(f"Classe prevista: {resultado}")
