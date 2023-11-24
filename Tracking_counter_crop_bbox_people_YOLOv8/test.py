# Importar las librerías necesarias
import cv2 # OpenCV es una librería de visión por computadora que permite procesar imágenes y videos
import pandas as pd # Pandas es una librería de análisis de datos que permite manipular tablas y series
import numpy as np # NumPy es una librería de computación científica que permite operar con matrices y vectores
from ultralytics import YOLO # YOLO es un modelo de detección de objetos que usa redes neuronales convolucionales
from tracker import * # Tracker es un módulo que contiene un algoritmo de seguimiento de objetos basado en el filtro de Kalman
from datetime import datetime # Datetime es un módulo que permite trabajar con fechas y horas
import os # Os es un módulo que permite interactuar con el sistema operativo

# Obtener la fecha y hora actual
now = datetime.now()

# Cargar el modelo YOLOv8 desde un archivo .pt
model=YOLO('yolov8s.pt')

# Definir una función que toma los parámetros del evento del mouse y muestra las coordenadas x e y del cursor
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE : # Si el evento es mover el mouse
        colorsBGR = [x, y] # Crear una lista con las coordenadas x e y
        print(colorsBGR) # Imprimir la lista en la consola

# Crear una ventana llamada RGB
cv2.namedWindow('RGB')
# Asignar la función RGB al evento del mouse en la ventana
cv2.setMouseCallback('RGB', RGB)

# Abrir el video peoplecount.mp4 como una fuente de captura
cap=cv2.VideoCapture('peoplecount.mp4')

# Abrir el archivo coco.txt como un archivo de lectura
my_file = open("coco.txt", "r")
# Leer el contenido del archivo y guardarlo en la variable data
data = my_file.read()
# Crear una lista con las clases de objetos del archivo, separadas por saltos de línea
class_list = data.split("\n")
# Inicializar un contador en cero
count=0
# Crear un objeto de la clase Tracker
tracker=Tracker()   
# Definir una lista de puntos que forman un polígono en la imagen
area=[(80,436),(41,449),(317,494),(317,470)] # RECTANGULO AZUL
# Definir un conjunto vacío para almacenar los identificadores de los objetos que entran al polígono
area_c=set()

# Definir una función que toma una imagen y la guarda en una carpeta con el nombre de la fecha y hora actual
def imgwrite(img):
    now = datetime.now() # Obtener la fecha y hora actual
    current_time = now.strftime("%d_%m_%Y_%H_%M_%S") # Formatear la fecha y hora como una cadena
    filename = '%s.png' % current_time # Crear el nombre del archivo con la cadena
    img_dir = os.path.join(os.getcwd(), "img") # Obtener la ruta de la carpeta img de forma automática
    if not os.path.exists(img_dir): # Verificar si la carpeta img existe o no
        os.makedirs(img_dir) # Crear la carpeta img si no existe
    cv2.imwrite(os.path.join(img_dir, filename), img) # Guardar la imagen en la ruta obtenida


def export_video(frames, output_file='video.mp4', fps=30):
    """
    Export frames as a video.

    Args:
    frames (list): List of frames to be exported.
    output_file (str): Name of the output video file.
    fps (int): Frames per second for the output video.

    Returns:
    None
    """
    # Define the codec for compression
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Get the height and width of the frames
    height, width = frames[0].shape[:2]

    # Create a VideoWriter object with the specified parameters
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Write each frame to the output video
    for frame in frames:
        out.write(frame)

    # Release the VideoWriter object and destroy all windows
    #out.release()
    #cv2.destroyAllWindows()

# Read and store the frames with the predictions and bbox
frames_bbox = []


# Iniciar un bucle infinito
while True:    
    # Leer un fotograma del video y guardarlo en la variable frame
    ret,frame = cap.read()
    # Si no se pudo leer el fotograma, salir del bucle
    if not ret:
        break
    # Incrementar el contador en uno
    count += 1
    # Si el contador es impar, continuar con el siguiente fotograma
    if count % 2 != 0:
        continue

    # Redimensionar el fotograma a un tamaño de 1020x500 píxeles
    frame=cv2.resize(frame,(1020,500))

    # Aplicar el modelo YOLOv8 al fotograma y obtener los resultados
    results=model.predict(frame)
 #   print(results)
    # Extraer las coordenadas de las cajas delimitadoras de los objetos detectados y guardarlas en una matriz
    a=results[0].boxes.data
    #print(a)
    # Convertir la matriz en un DataFrame de Pandas con valores flotantes
    px=pd.DataFrame(a).astype("float")
#    print(px)
    # Crear una lista vacía para almacenar las cajas delimitadoras de las personas
    list=[]
    # Recorrer cada fila del DataFrame
    for index,row in px.iterrows(): 
        # Obtener las coordenadas x e y de la esquina superior izquierda y la esquina inferior derecha de la caja delimitadora
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        # Obtener el índice de la clase del objeto detectado
        d=int(row[5])
        # Obtener el nombre de la clase del objeto detectado
        c=class_list[d]
        # Si el nombre de la clase es persona, agregar las coordenadas de la caja delimitadora a la lista
        if 'person' in c:
            list.append([x1,y1,x2,y2])
            
    # Aplicar el algoritmo de seguimiento a la lista de cajas delimitadoras y obtener los identificadores de los objetos
    bbox_idx=tracker.update(list)
    # Recorrer cada caja delimitadora con su identificador
    for bbox in bbox_idx:
        # Obtener las coordenadas x e y de la esquina superior izquierda y la esquina inferior derecha de la caja delimitadora y el identificador del objeto
        x3,y3,x4,y4,id=bbox
        # Comprobar si el punto (x4,y4) está dentro o sobre el polígono definido por la lista de puntos
        results=cv2.pointPolygonTest(np.array(area,np.int32),((x4,y4)),False)
        # Dibujar un rectángulo verde alrededor del objeto en el fotograma
        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
        # Dibujar un círculo magenta en el punto (x4,y4) en el fotograma
        cv2.circle(frame,(x4,y4),4,(255,0,255),-1)
        # Escribir el identificador del objeto en el fotograma con una fuente azul
        cv2.putText(frame,'Person: '+str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
        # Si el punto (x4,y4) está dentro o sobre el polígono
        if results>=0:
            # Recortar la imagen del objeto del fotograma
            crop=frame[y3:y4,x3:x4]
            # Guardar la imagen recortada en la carpeta img
            imgwrite(crop)
            # cv2.imshow(str(id),crop) 
            # Agregar el identificador del objeto al conjunto area_c
            area_c.add(id)
    # Dibujar un polígono rojo alrededor del área definida por la lista de puntos en el fotograma
    cv2.polylines(frame,[np.array(area,np.int32)],True,(100,50,255),3)
    # Imprimir el conjunto area_c en la consola
    print(area_c)
    # Obtener el tamaño del conjunto area_c, que representa el número de personas que han entrado al área
    k=len(area_c) # contador de personas que atraviesan la zona azul marcada
    # Escribir el número de personas en el fotograma con una fuente rojo
    cv2.putText(frame,str(k),(50,60),cv2.FONT_HERSHEY_PLAIN,5,(100,50,255),3)
    # Mostrar el fotograma en la ventana RGB
    cv2.imshow("RGB", frame)

    #print(str(k))

    # agregar los frames y reconstruir el video
    frames_bbox.append(frame)

    # Si se presiona la tecla Esc, salir del bucle
    if cv2.waitKey(1)&0xFF==27:
        break


# export the video
# Export the frames as a video
export_video(frames_bbox, output_file='output_video.mp4', fps=30)

# Liberar la fuente de captura
cap.release()
# Destruir todas las ventanas
cv2.destroyAllWindows()
