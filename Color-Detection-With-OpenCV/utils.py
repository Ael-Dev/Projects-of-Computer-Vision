import numpy as np
import cv2

def get_limits(color):
    # Insert the BGR to convert to HSV format
    c=np.uint8([[color]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    # Calculates the lower and upper limits of the color by subtracting and adding 10 to the hue value of the color,
    # respectively. The saturation and value values are set to 100.
    lowerLimit = hsvC[0][0][0] - 10, 100, 100 #10, 100, 100 amarillo BGR
    upperLimit = hsvC[0][0][0] + 10, 255, 255 #10, 255, 255

    lowerLimit = np.array(lowerLimit, dtype=np.uint8)
    upperLimit = np.array(upperLimit, dtype=np.uint8)

    return lowerLimit, upperLimit


