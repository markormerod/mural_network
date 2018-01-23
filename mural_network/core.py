import numpy as np

import glob
import os

import keras.backend as K
import keras.preprocessing.image as image_preprocessing
import keras.optimizers as optimizers
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from keras.utils.vis_utils import plot_model

from sklearn.utils import shuffle

import matplotlib.pyplot as plt
plt.style.use('ggplot')

class MuralNetwork():

    def __init__(self):
        self.vgg19_model = VGG19(weights='imagenet', include_top=False)
        self.preprocessed_X_train_file_path = '../data/preprocessed_X_train.npy'
        self.preprocessed_Y_train_file_path = '../data/preprocessed_Y_train.npy'
        self.preprocessed_X_test_file_path = '../data/preprocessed_X_test.npy'
        self.preprocessed_Y_test_file_path = '../data/preprocessed_Y_test.npy'

    def generate_training_data(self, X, Y, num_images=250, batch_size=10):

        image_generator = image_preprocessing.ImageDataGenerator(
            zoom_range=0.3,
            width_shift_range=0.3,
            height_shift_range=0.3)

        image_generator.fit(X)

        features_list = []
        labels_list = []
        total_batches = int(num_images/batch_size)
        batch_i = 0

        for X_batch, Y_batch in image_generator.flow(X, Y, batch_size=batch_size):
            batch_i += 1
            percent = batch_i/total_batches * 100

            print('Process image batch {} of {} ({}% complete) . . .'.format(batch_i, total_batches, percent))

            generated_images = preprocess_input(X_batch)
            features_list.append(self.vgg19_model.predict_on_batch(generated_images))
            labels_list.append(Y_batch)

            if len(features_list) * batch_size >= num_images:
                break

        X_features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        np.save(self.preprocessed_X_train_file_path, X_features)
        np.save(self.preprocessed_Y_train_file_path, labels)

        return X_features, labels

    def preprocess_and_save_test_data(self, X_test, Y_test):
        X_test = preprocess_input(X_test)
        X_test = self.vgg19_model.predict(X_test, batch_size=25)
        X_test = X_test.reshape(X_test.shape[0], -1)

        np.save(self.preprocessed_X_test_file_path, X_test)
        np.save(self.preprocessed_Y_test_file_path, Y_test)

        return X_test, Y_test

    def all_feature_files_exist(self):
        filepaths = [self.preprocessed_X_train_file_path,
                     self.preprocessed_Y_train_file_path,
                     self.preprocessed_X_test_file_path,
                     self.preprocessed_Y_test_file_path]

        return all([os.path.isfile(filepath) for filepath in filepaths])

    def load_features_from_file(self):
        X_train = np.load(self.preprocessed_X_train_file_path)
        X_train = X_train.reshape(X_train.shape[0], -1)
        Y_train = np.load(self.preprocessed_Y_train_file_path)

        X_test = np.load(self.preprocessed_X_test_file_path)
        X_test = X_test.reshape(X_test .shape[0], -1)
        Y_test = np.load(self.preprocessed_Y_test_file_path)

        return X_train, Y_train, X_test, Y_test

    def load_data(self, train_split=0.75, use_precomputed_features=True):
        if use_precomputed_features and self.all_feature_files_exist():
            return self.load_features_from_file()

        image_names = glob.glob('../images/[rl]*')
        images = [image_preprocessing.load_img(image_name, target_size=(224, 224)) for image_name in image_names]
        images = shuffle(images)

        X = np.array([image_preprocessing.img_to_array(image) for image in images])

        image_class = lambda x: 0 if x == 'l' else 1
        image_prefix = '../images/'
        image_class_position = len(image_prefix)
        Y_labels = np.array([image_class(x[image_class_position]) for x in image_names])

        Y_one_hot = K.get_session().run(K.one_hot(Y_labels, 2))

        # Seperate test data and generate new samples from training data
        train_data_size = int(X.shape[0] * train_split)

        Y_test = Y_one_hot[train_data_size:]
        X_test, Y_test = self.preprocess_and_save_test_data(X[train_data_size:], Y_test)

        X_train = X[0:train_data_size]
        Y_train = Y_one_hot[0:train_data_size]
        X_train, Y_train = self.generate_training_data(X_train, Y_train)

        X_train = X_train.reshape(X_train.shape[0], -1)

        return X_train, Y_train, X_test, Y_test

    def get_model(self, input_length, learning_rate=0.0001):
        model = Sequential()

        model.add(Dense(units=512, input_dim=input_length, activation='linear', kernel_initializer='glorot_normal'))
        model.add(LeakyReLU(alpha=0.33))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())

        model.add(Dense(units=512, activation='linear', kernel_initializer='glorot_normal'))
        model.add(LeakyReLU(alpha=0.33))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())

        model.add(Dense(units=64, activation='linear', kernel_initializer='glorot_normal'))
        model.add(LeakyReLU(alpha=0.33))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())

        model.add(Dense(units=2, activation='softmax'))

        optimizer = optimizers.Adagrad(lr=learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        if not os.path.exists('../plot'):
            os.makedirs('../plot')
        plot_model(model, to_file='../plot/model.png', show_shapes=True)

        print(model.summary())

        return model

mural_network = MuralNetwork()
X_train, Y_train, X_test, Y_test = mural_network.load_data()

print("Loyalist proportion: {}".format(sum(Y_test[:,0]==0)/len(Y_test)))

learning_rate = 0.00001
model = mural_network.get_model(X_train.shape[1], learning_rate)

epochs = 400
batch_size = 32
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=batch_size)

# Plot the loss for the training and validation sets over the training epochs
plt.clf()
plt.plot(history.history['loss'], color='slateblue')
plt.plot(history.history['val_loss'], color='orange')
plt.title('Model Loss over epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set loss', 'Validation set loss'])
plt.savefig('../plot/neural_network_loss')

model.save('../data/model.h5')
