import cv2
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation
from keras.callbacks import TensorBoard
from keras.utils import to_categorical

from random import shuffle

PATH_TO_DATA = 'data/blue_data/'

label_dic = {'soldier': 0, 'genji': 1, 'reaper': 2, 'ana': 3, 'bastion': 4, 'brig': 5, 'doomfist': 6, 'dva': 7, 'hanzo':8,
'junkrat': 9, 'lucio': 10, 'mccree': 11, 'mei': 12, 'mercy': 13, 'moira': 14, 'orisa': 15, 'pharah': 16, 'reinhardt': 17,
'roadhog': 18, 'sombra': 19, 'symmetra': 20, 'torbjorn': 21, 'tracer': 22, 'widow': 23, 'winston': 24, 'zarya': 25,
'zenyatta': 26, 'hammond': 27, 'ashe': 28, 'baptiste': 29, 'unknownhero': 30}

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
            y.append(label_dic[label])
    #Convert class vector to binary class matrix
    y = keras.utils.to_categorical(y, len(label_dic))

    #Shuffle the order in which the dataset is gotten
    zipped = list(zip(X, y))
    shuffle(zipped)
    X, y = zip(*zipped)
    return X, y

def get_model():
    model = Sequential([
        Conv2D(32, (3,3), input_shape=(25, 30, 3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(32, (3,3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(32, (3,3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128),
        Activation('relu'),
        Dropout(0.5),
        Dense(64),
        Activation('relu'),
        Dropout(0.5),
        Dense(31),
        Activation('softmax')
    ])
    return model

def read_vod_load_right_for_model(X_batch, resize_to_720P=True, train = False):
    if train:
        X_loaded = []
        print("Total number of JPGS: " + str(len(X_batch)))
        print(X_batch[0:5])
        for path in X_batch:
            num_loaded = len(X_loaded)
            if num_loaded % 100 == 0:
                print("Right Side: Processed " + str(num_loaded)+ " images already")
            img = cv2.imread(path)
            img = img[49:74, 860:890]
            # dividing by 255 leads to faster convergence through normalization.
            X_loaded.append(np.array(img)/(255))
        cv2.imwrite('sample.jpg', X_loaded[0] * 255)
        return X_loaded
    else:
    # X_batch is just a bunch of file names. We need to load the image to pass it to a net!
        X7_loaded = []
        X8_loaded = []
        X9_loaded = []
        X10_loaded = []
        X11_loaded = []
        X12_loaded = []
        X_loaded_arr = [X7_loaded, X8_loaded, X9_loaded, X10_loaded, X11_loaded, X12_loaded]

        for path in X_batch:
            img = cv2.imread(path)
            #Crop out player 1
            img_7 = img[49:74, 860:890]
            img_8 = img[49:74, 931:961]
            img_9 = img[49:74, 1001:1031]
            img_10 = img[49:74, 1072:1102]
            img_11 = img[49:74, 1143:1173]
            img_12 = img[49:74, 1214:1244]
            # Resize image
            img_array = [img_7, img_8, img_9, img_10, img_11, img_12]
            for index, image in enumerate(img_array):
                #image = cv2.resize(image, (74,66)

                # dividing by 255 leads to faster convergence through normalization.
                X_loaded_arr[index].append(np.array(image)/(255))

        cv2.imwrite('sample7.jpg', X_loaded_arr[0][0] * 255)
        cv2.imwrite('sample8.jpg', X_loaded_arr[1][0] * 255)
        cv2.imwrite('sample9.jpg', X_loaded_arr[2][0] * 255)
        cv2.imwrite('sample10.jpg', X_loaded_arr[3][0] * 255)
        cv2.imwrite('sample11.jpg', X_loaded_arr[4][0] * 255)
        cv2.imwrite('sample12.jpg', X_loaded_arr[5][0] * 255)

        return [X7_loaded, X8_loaded, X9_loaded, X10_loaded, X11_loaded, X12_loaded]

def train():
    X, y = get_dataset(PATH_TO_DATA)
    """
    #80% of the data will be in our training set
    X_train = np.asarray(read_vod_load_red_for_model(X[0:int(len(X) * 0.8)], train = True))
    y_train = np.asarray(y[0:int(len(X) * 0.8)])

    #20% of the data will be in our validation set
    X_val = np.asarray(read_vod_load_red_for_model(X[int(len(X) * 0.8):], train = True))
    y_val = np.asarray(y[int(len(X) * 0.8):])
    """
    X_train = np.asarray(read_vod_load_right_for_model(X, train = True))
    y_train = np.asarray(y)

    model = get_model()
    model.compile(loss='categorical_crossentropy',
                  optimizer = keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy'])

    model.summary()
    num_epochs = 54
    batch_size = 16

    print("Beginning training!")
    print("Training set is size %d and Val set is size %d" % (int(len(X) * 0.8), int(len(X) * 0.2)))

    #model_hist = model.fit(x = X_train, y = y_train, batch_size = batch_size, epochs = num_epochs, verbose = 1,validation_data = (X_val, y_val))
    model_hist = model.fit(x = X_train, y = y_train, batch_size = batch_size, epochs = num_epochs, verbose = 1)
    model.save('right_hero_model.h5')
    print("Training complete!")

    # summarize history for accuracy
    plt.plot(model_hist.history['acc'])
    #lt.plot(model_hist.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    #plt.legend(['train', 'validation'], loc='upper left')
    plt.legend(['train'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(model_hist.history['loss'])
    #plt.plot(model_hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    #plt.legend(['train', 'validation'], loc='upper left')
    plt.legend(['train'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    train()
