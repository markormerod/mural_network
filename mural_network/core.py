import numpy as np

import glob
from os import path

import keras.preprocessing.image as image_preprocessing
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input

def get_vgg19_predictions():
    vgg19_features_file_path = '../data/vgg19_features.npy'

    if path.isfile(vgg19_features_file_path):
        return np.load(vgg19_features_file_path)

    image_names = glob.glob('../images/*')
    images = [image_preprocessing.load_img(image_name, target_size=(224, 224)) for image_name in image_names]

    X = np.array([image_preprocessing.img_to_array(image) for image in images])

    vgg19_model = VGG19(weights='imagenet', include_top=False)

    X = preprocess_input(X)
    X_features = vgg19_model.predict(X, batch_size=16)

    np.save(vgg19_features_file_path, X_features)

    return X_features

def load_data(train_split=0.15):
    X_vgg19_features = get_vgg19_predictions()

    image_class = lambda x: 0 if x == 'l' else (1 if x == 'r' else 2)


    X_train = None
    Y_train = None
    X_test = None
    Y_test = None

    return X_train, Y_train, X_test, Y_test

def get_model():
    model = None

    return model

X_train, Y_train, X_test, Y_test = load_data()

model = get_model()
