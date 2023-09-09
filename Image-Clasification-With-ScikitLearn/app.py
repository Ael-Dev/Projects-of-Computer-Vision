import os
import cv2
import pickle
import numpy as np
from skimage.io import imread
from skimage.transform import resize

def read_images(path_directory):
    images = []
    for name in os.listdir(path_directory):
        img = imread(os.path.join(path_directory, name))
        img = resize(img, (15,15))
        img_array = img.flatten()
        images.append(img_array)
    return np.asarray(images)


def predict(val):
    if val == 0:
        label = 'empty'
        #color = (0, 255, 0)  # Green
    else:
        label = 'not_empty'
        #color = (0, 0, 255)  # Red
    return label #, color



if __name__ == '__main__':
    # Load the saved SVM model from the pkl file
    svm_model = pickle.load(open('model.pkl','rb'))
    path = "./data/pred_data/"
    img_pred_array = read_images(path)

    # make a predictions for new data
    y_pred = svm_model.predict(img_pred_array)
    for result in y_pred:
       print(f"PREDICTION: {result}")
