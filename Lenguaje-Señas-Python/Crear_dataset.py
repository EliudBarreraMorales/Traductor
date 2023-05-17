import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
'''
este código está procesando las posiciones de las manos y sus landmarks 
y preparando los datos para su uso en un modelo de aprendizaje automático, 
donde la dirección es la etiqueta asociada con cada conjunto de landmarks de la mano
'''
mp_hands = mp.solutions.hands
'''
Este módulo proporciona la solución de MediaPipe Hands, 
que es un modelo pre-entrenado para detectar y 
reconocer las posiciones de las manos en imágenes y videos
'''
mp_drawing = mp.solutions.drawing_utils
'''
Este módulo proporciona una variedad de utilidades de dibujo para ayudar a 
visualizar los resultados de la detección de landmarks. Estas utilidades 
permiten dibujar las líneas de conexión entre los landmarks de las manos, 
el círculo alrededor de los landmarks detectados y la dirección de los dedos.
'''
mp_drawing_styles = mp.solutions.drawing_styles
'''
: Este módulo proporciona una variedad de estilos de dibujo 
predefinidos que se pueden utilizar, para personalizar la apariencia de las visualizaciones 
de detección de landmarks
'''
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'# en la carpeta llamada data

# se crean 2 listas
data = []
labels = []
'''
Se recorre un directorio que contiene subdirectorios de datos (imágenes) 
y procesa cada imagen individualmente:
DATA_DIR =  directorio raiz
dir_ = subdirectorio
img_path = nombre del archivo de la imagen actual

'''
for dir_ in os.listdir(DATA_DIR):# se recorre los elementos (subdirectorios) en el directorio especificado
    # La variable dir_ es una variable de iteración que tomará el valor de cada subdirectorio encontrado en el bucle.
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):# img_path: es una variable de iteración que tomará el valor de cada archivo de imagen encontrado en el bucle.
        data_aux = []# es una lista temporal para datos temporales
        x_ = []
        y_ = []
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))# se lee la imagen pasandole la raiz, el subdirectorio y la imagen
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)# convertir el color de la imagen a RGB

        results = hands.process(img_rgb)# aqui alamaceno la salida despues de procesar la imagen
        if results.multi_hand_landmarks:# Luego se verifica si "results.multi_hand_landmarks" existe, 
            # lo que significa que al menos una mano ha sido detectada en la imagen procesada
            for hand_landmarks in results.multi_hand_landmarks:# Si una o más manos han sido detectadas, el código recorre todos los landmarks de la mano 
                for i in range(len(hand_landmarks.landmark)): # para cada landmark se almacena su posición en x
                    # se almacenan en su lista correspondiente tanto en y como en x
                    x = hand_landmarks.landmark[i].x 
                    y = hand_landmarks.landmark[i].y
                    
                    x_.append(x)# Almacenamos el valor de x
                    y_.append(y)# Almacenamos el valor de y
# Luego, se calculan las coordenadas relativas de cada landmark en relación con el punto más cercano 
# en el borde superior izquierdo de la imagen, y se almacenan en la lista data_aux.
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)
# dir_ : es simplemente un nombre de variable que puede ser utilizado para representar una dirección o etiqueta

f = open('data.pickle', 'wb')# write binary
pickle.dump({'data': data, 'labels': labels}, f)
f.close()