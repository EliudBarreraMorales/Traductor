import pickle
import cv2
import mediapipe as mp
import numpy as np

'''
Pickle es un formato de serialización binario, es especificamente utilizado en Python:
Consiste en convertir un objeto de Python (normalmente una lista o diccionario) en un string
'''

# pickle.load es para cargar el archivo y poder leerlo
model_dict = pickle.load(open('./model.p', 'rb'))# rb = read-binary
model = model_dict['model']

cap = cv2.VideoCapture(0)# Abro la camara
'''
hand landmarks = puntos de referencia de la mano
'''
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.3)
'''
- static_image_mode: que se establece en True, 
lo que significa que el modelo se ejecutará en modo de imagen estática en lugar de video en tiempo real
- min_detection_confidence se establece en 0.3, 
lo que significa que las detecciones de manos con una confianza inferior a este valor no serán reportadas
'''
# diccionario para almecenar las letras
# 0: 'A', 1: 'B',  2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',  10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
labels_dict = { 0: 'A', 1: 'B',  2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',  10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z' }
'''
- Inicializa la detección de manos 
- Captura los fotogramas de video en un bucle infinito.
- Si se detectan manos en el fotograma, dibuja los landmarks de las manos y los conecta con líneas para visualización. Luego, extrae las coordenadas x e y de los landmarks y las almacena en una lista. Las coordenadas se normalizan restando la coordenada mínima de cada eje.
- Utiliza las coordenadas normalizadas para crear un vector de características para esa mano.
- Utiliza el modelo previamente entrenado para predecir el carácter de lenguaje de señas que se está realizando.
- Dibuja un rectángulo alrededor de la mano y muestra el carácter clasificado en el fotograma.
- Espera a que se presione la tecla "q" para salir del programa.
'''
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)# convertir el color de BGR a RGB

    results = hands.process(frame_rgb)# almeceno la salida
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                # parametros
                frame, 
                hand_landmarks,  # modelo de salida
                mp_hands.HAND_CONNECTIONS,  # para dibujar las líneas de conexión entre los landmarks de las manos
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()