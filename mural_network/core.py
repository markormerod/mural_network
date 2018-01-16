import numpy as np

import glob
import os

import keras.backend as K
import keras.preprocessing.image as image_preprocessing
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.vis_utils import plot_model

from sklearn.utils import shuffle

def get_vgg19_predictions(X):
    vgg19_features_file_path = '../data/vgg19_features.npy'

    if os.path.isfile(vgg19_features_file_path):
        return np.load(vgg19_features_file_path)

    vgg19_model = VGG19(weights='imagenet', include_top=False)

    X = preprocess_input(X)

    X_features = vgg19_model.predict(X, batch_size=16)

    np.save(vgg19_features_file_path, X_features)

    return X_features

def load_data(train_split=0.85):
    image_names = glob.glob('../images/[rl]*')
    images = [image_preprocessing.load_img(image_name, target_size=(224, 224)) for image_name in image_names]

    X = np.array([image_preprocessing.img_to_array(image) for image in images])

    X_vgg19_features = get_vgg19_predictions(X)
    X_vgg19_features = X_vgg19_features.reshape(X_vgg19_features.shape[0], -1)

    image_class = lambda x: 0 if x == 'l' else 1
    image_prefix = '../images/'
    image_class_position = len(image_prefix)
    Y_labels = np.array([image_class(x[image_class_position]) for x in image_names])

    Y_one_hot = K.get_session().run(K.one_hot(Y_labels, 2))

    X_vgg19_features, Y_train_one_hot_shuffled = shuffle(X_vgg19_features, Y_one_hot)

    train_data_size = int(X_vgg19_features.shape[0] * train_split)

    X_train = X_vgg19_features[0:train_data_size]
    Y_train = Y_one_hot[0:train_data_size]
    X_test = X_vgg19_features[train_data_size:]
    Y_test = Y_one_hot[train_data_size:]

    return X_train, Y_train, X_test, Y_test

def get_model(input_length):
    model = Sequential()

    model.add(Dense(units=128, activation='relu', input_dim=input_length))
    model.add(Dropout(0.4))

    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(units=16, activation='relu'))
    model.add(Dropout(0))

    model.add(Dense(units=2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if not os.path.exists('../plot'):
        os.makedirs('../plot')
    plot_model(model, to_file='../plot/model.png', show_shapes=True)

    print(model.summary())

    return model

X_train, Y_train, X_test, Y_test = load_data()

model = get_model(X_train.shape[1])

epochs = 100
batch_size = 32
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=batch_size)