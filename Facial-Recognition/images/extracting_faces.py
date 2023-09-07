import cv2
import os

images_path = "./input_images"

# crear una carpeta para guardar los rostros
if not os.path.exists("faces"):
    os.makedirs("faces")
    print("carpeta >> faces << creada")

# deteccion de rostros
faceClassification = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

counter = 0

# recuperar el nombre de cada imagen contenida dentro de una determinada carpeta
for ImgName in os.listdir(images_path):
    image_path = os.path.join(images_path, ImgName)
    # leer imagen de la ruta definida
    image = cv2.imread(image_path)
    # aplicar el detector facial sobre cada imagen
    # para obtener la informacion de la ubicacion de los rostros
    faces = faceClassification.detectMultiScale(image, 1.1, 5)
    # acceder a la informacion obtenida
    for (x, y, w, h) in faces: # izq, der, ancho y alto
        # dibujar el bbox
        #cv2.rectangle(image, (x,y),(x+w,y+h), (0,255,0),3)
        # guardar en una variable las coordenadas del rostro de cada imagen
        face = image[y:y+h, x:x+w]
        # redimensionar para que las imagenes tengan el mismo ancho y alto
        face = cv2.resize(face, (160,160))
        # guardar la imagen del rostro redimensionado
        cv2.imwrite("faces/"+str(counter)+ ".jpg", face)
        counter += 1
        	
        #cv2.imshow("image", face)
        #cv2.waitKey(0)
    # visualizar la imagen
    #cv2.imshow("image", image)
    #cv2.waitKey(0)

#cv2.destroyAllWindows()
