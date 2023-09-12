import cv2
from ultralytics import YOLO
from sort import Sort
import numpy as np



if __name__ == '__main__':
    # leer el video con opencv
    path_video = "./data/people_walking.mp4"
    cap = cv2.VideoCapture(path_video)

    # ----------------------------------------------------------------
    # carcar el modelo
    model_path = "./yolov8n.pt"
    model = YOLO(model_path)
    # ----------------------------------------------------------------
    # Inicializar el tracker
    tracker = Sort()
    
    # ----------------------------------------------------------------


    while cap.isOpened():
        status, frame = cap.read()

        # sino hay nada salir 
        if not status:
            break

        # ----------------------------------------------------------------
        # usar el modelo
        results = model(frame, stream=True) # Aislar c/u de las predicciones con stream=True

        # acceder a los bbox
        for res in results:
            # agregar un filtro para graficar aquellos superior al umbral definido para eliminar los bbox del fondo
            filtered_indices = np.where(res.boxes.conf.cpu().numpy() > 0.5)[0] 
            # extraemos las coordenadas de los bbox
            boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int) # agregandole el filtro [filtered_indices]
            # pasamos la matriz al tracker
            tracks = tracker.update(boxes) # el tracker le va agregar a mi matriz una columna ID del obj
            tracks = tracks.astype(int)

            # dibujar los bbox con cv2
            for xmin, ymin, xmax, ymax, track_id in tracks:
                score = res.boxes.conf.cpu().numpy()[0]*100
                score = score.astype(int)
                cv2.putText(img=frame, text=f"Id: {track_id} score:{score}%", org=(xmin, ymin-10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0,255,0), thickness=2)
                cv2.rectangle(img=frame, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 255, 0), thickness=3)
   
        # plotear los resultados con los bbox por defecto del modelo
        #frame = results[0].plot() # se puede crear nuestros propios bbox
        # ----------------------------------------------------------------
        # ahora teniendo los resultados tenemos que pasarle al algoritmo sort
        # sort funciona con bbox

        # mostrar los frames
        cv2.imshow("frame", frame)

        # finalizar lectura de video
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()