import cv2
import os
import face_recognition

# ------------------------------------------------------------------------
# activando la gpu
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)

# -------------------------------------------------------------------------
# codigicar las rostros 
faces_path = "./images/faces"

# listas para almacenar la informacion obtenida de cada rostro
faces_encodings = []
faces_labels = []

for name in os.listdir(faces_path):
    # establecer la ruta de cada imagen
    image_path = os.path.join(faces_path, name)
    image = cv2.imread(image_path) # leer imagen de la ruta
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # pasar a de BGR -> RGB

    # codificar c/u imagenes -> retornara un vector codificado de 128 elementos
    # dado que ya tenemos el rostro detectado, se usrá known_face_locations
    # pasar el tamaño de la imagen con la que redimensionamos
    face_encoding = face_recognition.face_encodings(image, known_face_locations=[(0,160,160,0)])[0]
    # guardar la informacion obtenida
    faces_encodings.append(face_encoding)
    faces_labels.append(name.split('.')[0])

# -------------------------------------------------------------------------
# aplicar sobre un video streaming
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# deteccion de rostros
faceClassification = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if ret == False: break
    frame = cv2.flip(frame, 1)
    original = frame.copy() # guardar la imagen original
    
    faces = faceClassification.detectMultiScale(frame, 1.1, 5)
    for (x, y, w, h) in faces: # izq, der, ancho y alto
        # guardar en una variable las coordenadas del rostro de cada imagen original
        face = original[y:y+h, x:x+w]

        # ----------------------------------------------------------------
        # convertir de bgr a rgb
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        # codificar c/u imagenes -> retornara un vector codificado de 128 elementos
        current_face_encoding = face_recognition.face_encodings(face, known_face_locations=[(0,w,h,0)])[0]
        # comparar rostros y asignar etiqueta -> retornara un vector de valores booleanos
        # si hay alguno que coincida cambiara a true algun elemento del vector
        result = face_recognition.compare_faces(faces_encodings, current_face_encoding)
        if True in result:
            index = result.index(True) # recuperar la posicion de donde apareció True
            name = faces_labels[index]
            color = (100,200,0)
        else:
            name = "Unknown face"
            color = (255,255,255)
        # ----------------------------------------------------------------
        # dibujar el bbox y el nombre
        cv2.rectangle(frame, (x, y+h),(x+w, y+h+25), color, -1)
        cv2.rectangle(frame, (x, y),(x+w, y+h), color,3)
        cv2.putText(frame, name, (x, y+h + 25),2,1, (255,255,0), 2, cv2.LINE_AA)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
cap.release()
cv2.destroyAllWindows()







