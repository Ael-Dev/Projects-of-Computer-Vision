import cv2
from utils import get_limits
from PIL import Image
# para cuando en la funcion get limits = 10, 100, 100
# yellow:  [0,255,255] 
# green: [100,255,100]
# purpura: [255,20,255] 
# azul: [255,20,20]
# rojo: [150,50,255]
def obtener_rgb_color(color):
    bgr = [0,0,0] # black
    if color=="yellow":
        bgr = [0,255,255]
    elif color == "green":
        bgr = [50,255,50]
    elif color == "red":
        bgr = [150,50,255]
    elif color == "blue":
        bgr = [255,30,30]
    else:
        bgr = [255,255,255] # white
    return bgr

def deteccion_color(color):

    color_detect = obtener_rgb_color(color)#[0,255,255]

    cap = cv2.VideoCapture(0) # 0 default webcam
    while True:
        # start an infinite loop that reads frames from the camera
        ret, frame = cap.read()

        #if not ret:
        #    continue

        # convert to BGR to HSV format
        hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # obtain the limits
        lowerLimit, upperLimit = get_limits(color=color_detect)

        # determine the pixels of the object location
        mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

        # bounding box- converto to object PIL
        mask_ = Image.fromarray(mask) 
        # obtain the bounding box if detect an object
        bbox = mask_.getbbox()
        if bbox is not None:
            x1,y1,x2,y2 = bbox # obtain the coordinates
            # draw a rectangle
            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 5)
        

        # display the captured frames
        cv2.imshow('frame', mask) # .mask)frame

        # wait for the user to press the 'q' key to exit the loop using
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    deteccion_color("green")