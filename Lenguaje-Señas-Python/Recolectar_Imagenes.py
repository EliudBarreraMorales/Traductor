import os
import cv2

# Si no existe una carpeta llamada data, creala
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 10  # 25 letras del abecedario
dataset_size = 100 # num de carpetas que contiene imagenes

'''
Utiliza la biblioteca OpenCV para capturar 
imágenes desde la cámara en vivo y guardarlas en subdirectorios en el disco
'''
cap = cv2.VideoCapture(0)# el 0 es que especifica que se debe utilizar la cámara predeterminada del sistema
'''
crea subdirectorios en el directorio de datos especificado por 
DATA_DIR para cada clase etiquetada por su índice j. 
Si un subdirectorio para la clase actual no existe, 
entonces se crea utilizando la función os.makedirs().
'''
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Recolectando informacion para la clase: {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()# se muestra la camara en vivo en una ventana
        cv2.putText(frame, 'Listo? Presiona "Q"', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)# con la letra Q se presiona para capturar las imagenes
        #flip0 = cv2.flip(frame,1)
        #cv2.imshow("camara", flip0)
        cv2.imshow("camara", frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter <= dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)# Cada imagen se guarda en disco en el subdirectorio correspondiente utilizando la función cv2.imwrite()
        counter += 1

cap.release()# el objeto videocapture se libera
cv2.destroyAllWindows()# se cierran todas las ventanas 