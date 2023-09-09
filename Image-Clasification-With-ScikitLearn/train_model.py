# import libraries
# ------------------------------------------------------
import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV#, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import pickle


# prepare data
# ------------------------------------------------------
def data_preprocessing(input_path_directory, categories):
    data = []
    labels = []
    for index_category, category in enumerate(categories): # iterate over the main directory
        for name in os.listdir(os.path.join(input_path_directory, category)): # path for every category
            img_path = os.path.join(input_path_directory, category, name) # image path
            img = imread(img_path)
            # Note: all the images should have the same size
            img = resize(img, (15,15))
            # convert image to vector
            data.append(img.flatten())
            labels.append(index_category)
    # convert vectors as numpy arrays
    data = np.asarray(data)
    labels = np.asarray(labels)

    return data, labels


# train and test split
# ------------------------------------------------------
def train_test_split_data(X, y,size, random):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, shuffle=True, stratify=y, random_state=random)
    return X_train, X_test, y_train, y_test


# ------------------------------------------------------
# train model
def train_model(X_train, y_train, model, params):
    gs = GridSearchCV(model, params)
    gs.fit(X_train, y_train)
    # obtain the best model
    best_model = gs.best_estimator_
    return best_model

# test performance
# ------------------------------------------------------
def test_model_performance(X, y, model):
    y_pred = model.predict(X)
    score = accuracy_score(y_pred, y)
    print(classification_report(y, y_pred))
    return score

# save the model
# ------------------------------------------------------ 
def save_model(model, name):
    pickle.dump(model, open(f'./{name}.pkl', 'wb'))


# main program
# ------------------------------------------------------
if __name__ == "__main__":
    # ------------------------------------------------------
    # prepare data
    input_path_directory = "./data"
    categories = ["empty", "not_empty"] # 0: empty, 1: not_empty
    data, labels = data_preprocessing(input_path_directory, categories)

    # ------------------------------------------------------
    # split data
    X_train, X_test, y_train, y_test = train_test_split_data(data, labels, 0.2, 42)

    # ------------------------------------------------------
    # define the model
    model_classifier = SVC()

    # ------------------------------------------------------
    # train model
    params = [
        {'gamma':[0.01,0.001,0.0001],
            'C':[1,10,100,1000]}
    ]
    best_model = train_model(X_train, y_train, model_classifier, params)

    # ------------------------------------------------------
    # evaluate the model
    accuracy_score_train = test_model_performance(X_train, y_train, best_model)
    accuracy_score_test = test_model_performance(X_test, y_test, best_model)

    # show the results
    print(f"ACCURACY TRAIN SCORE: {accuracy_score_train}")
    print(f"ACCURACY TEST SCORE: {accuracy_score_test}")

    if accuracy_score_test > 0.9:
        save_model(best_model, 'model')
        print("best model found")
    else:
        print("best model not found")
    