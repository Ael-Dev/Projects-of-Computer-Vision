import cv2
from utils import get_limits
from PIL import Image
# para cuando en la funcion get limits = 10, 100, 100
# yellow:  [0,255,255] 
# green: [100,255,100]
# purpura: [255,20,255] 
# azul: [255,20,20]
# rojo: [150,50,255]

color_detect = [100,255,100] 
cap = cv2.VideoCapture(0) # number of webcams to capture
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
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 5)
    

    # display the captured frames
    cv2.imshow('frame', frame) # .mask)

    # wait for the user to press the 'q' key to exit the loop using
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
