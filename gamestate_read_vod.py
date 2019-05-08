import cv2
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import imutils
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Input, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from random import shuffle

PATH_TO_DATA = 'data/gamestate_data/'
gamestate_dic = {'lobby':1 , 'ingame': 0}

"""## **Create a Model based on Converted VODS**"""
def get_dataset(data_root_path, get_test_set=False):
    X = []
    y = []
    for folder in os.listdir(data_root_path):
        path_to_images = data_root_path + folder
        label = folder.split("_")[0]

        if not os.path.isdir(path_to_images):
            continue
        # skip when "test" is in folder and we don't want the test set
        if "test" in folder and not get_test_set:
            continue
        # skip when "test" in not in folder and we want the test set.
        if "test" not in folder and get_test_set:
            continue

        for image_name in os.listdir(path_to_images):
            if ".jpg" not in image_name:
                continue
            #Put image paths into the array
            X.append(path_to_images + "/" + image_name)
            #Put the value in the dictionary
            y.append(gamestate_dic[label])
    #Convert class vector to binary class matrix
    y = keras.utils.to_categorical(y, len(gamestate_dic))

    #Shuffle the order in which the dataset is gotten
    zipped = list(zip(X, y))
    shuffle(zipped)
    X, y = zip(*zipped)
    return X, y

def get_model():
    input_shape = (40,40, 3)
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation=tf.nn.softmax))
    return model

def gamestate_load_images_for_model(X_batch, resize_to_720P=True, train = False):

    X_loaded = []
    for path in X_batch:
        img = cv2.imread(path)
        num_loaded = len(X_loaded)
        if num_loaded % 100 == 0:
            print("Gamestate: Processed " + str(num_loaded)+ " images already")
        if np.size(img, 0) != 720:
            img = imutils.resize(img, width=1280)
        img = img[20:60, 962:1002]
        # dividing by 255 leads to faster convergence through normalization.
        X_loaded.append(np.array(img)/(255))
    cv2.imwrite('lobby_sample.jpg', X_loaded[0] * 255)
    return X_loaded

def train():
    X, y = get_dataset(PATH_TO_DATA)

    #80% of the data will be in our training set
    X_train = np.asarray(gamestate_load_images_for_model(X[0:int(len(X) * 0.8)], train = True))
    y_train = np.asarray(y[0:int(len(X) * 0.8)])

    #20% of the data will be in our validation set
    X_val = np.asarray(gamestate_load_images_for_model(X[int(len(X) * 0.8):], train = True))
    y_val = np.asarray(y[int(len(X) * 0.8):])

    model = get_model()
    model.compile(loss='categorical_crossentropy',
                  optimizer = keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy'])

    model.summary()
    num_epochs = 10
    batch_size = 16

    print("Beginning training!")
    print("Training set is size %d and Val set is size %d" % (int(len(X) * 0.8), int(len(X) * 0.2)))

    model_hist = model.fit(x = X_train, y = y_train, batch_size = batch_size, epochs = num_epochs, verbose = 1, validation_data = (X_val, y_val))
    model.save('gamestate_model.h5')
    print("Training complete!")

    # summarize history for accuracy
    plt.plot(model_hist.history['acc'])
    plt.plot(model_hist.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(model_hist.history['loss'])
    plt.plot(model_hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    train()
