import numpy as np

import glob
from os import path

import keras.backend as K
import keras.preprocessing.image as image_preprocessing
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input

from sklearn.utils import shuffle

def get_vgg19_predictions(X):
    vgg19_features_file_path = '../data/vgg19_features.npy'

    if path.isfile(vgg19_features_file_path):
        return np.load(vgg19_features_file_path)

    vgg19_model = VGG19(weights='imagenet', include_top=False)

    X = preprocess_input(X)
    X_features = vgg19_model.predict(X, batch_size=16)

    np.save(vgg19_features_file_path, X_features)

    return X_features

def load_data(train_split=0.85):
    image_names = glob.glob('../images/*')
    images = [image_preprocessing.load_img(image_name, target_size=(224, 224)) for image_name in image_names]

    X = np.array([image_preprocessing.img_to_array(image) for image in images])

    X_vgg19_features = get_vgg19_predictions(X)

    image_class = lambda x: 0 if x == 'l' else (1 if x == 'r' else 2)
    image_prefix = '../images/'
    image_class_position = len(image_prefix)
    Y_labels = np.array([image_class(x[image_class_position]) for x in image_names])

    with K.get_session() as sess:
        Y_one_hot = sess.run(K.one_hot(Y_labels, 3))

    X_vgg19_features, Y_train_one_hot_shuffled = shuffle(X_vgg19_features, Y_one_hot)

    train_data_size = int(X_vgg19_features.shape[0] * train_split)

    X_train = X_vgg19_features[0:train_data_size]
    Y_train = Y_one_hot[0:train_data_size]
    X_test = X_vgg19_features[train_data_size:]
    Y_test = Y_one_hot[train_data_size:]

    return X_train, Y_train, X_test, Y_test

def get_model():
    model = None

    return model

X_train, Y_train, X_test, Y_test = load_data()

model = get_model()
