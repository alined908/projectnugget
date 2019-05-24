import cv2
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Input, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from random import shuffle
PATH_TO_DATA = 'data/yellow_ult_data/'
ult_charge_dic = {'uncharged':0 , 'charged': 1}

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
            y.append(ult_charge_dic[label])
    #Convert class vector to binary class matrix
    y = keras.utils.to_categorical(y, len(ult_charge_dic))

    #Shuffle the order in which the dataset is gotten
    zipped = list(zip(X, y))
    shuffle(zipped)
    X, y = zip(*zipped)
    return X, y

def get_model():
    input_shape = (25,21, 3)
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation=tf.nn.softmax))
    return model

def left_ult_charge_load_images_for_model(X_batch, resize_to_720P=True, train = False):
    if train:
        X_loaded = []
        for path in X_batch:
            img = cv2.imread(path)
            img = img[49:74, 39:60]
            # dividing by 255 leads to faster convergence through normalization.
            X_loaded.append(np.array(img)/(255))
        cv2.imwrite('sample.jpg', X_loaded[0] * 255)
        return X_loaded

    else:
        # X_batch is just a bunch of file names. We need to load the image to pass it to a net!
        X1_loaded = []
        X2_loaded = []
        X3_loaded = []
        X4_loaded = []
        X5_loaded = []
        X6_loaded = []

        X_loaded_arr = [X1_loaded, X2_loaded, X3_loaded, X4_loaded, X5_loaded, X6_loaded]
        for path in X_batch:
            num_loaded = len(X1_loaded)
            if num_loaded % 100 == 0:
                print("Left Ults: Processed " + str(num_loaded)+ " images already")
            img = cv2.imread(path)
            #Crop out player 1
            img_1 = img[49:74, 39:60]
            img_2 = img[49:74, 110:131]
            img_3 = img[49:74, 181:202]
            img_4 = img[49:74, 252:273]
            img_5 = img[49:74, 323:344]
            img_6 = img[49:74, 394:415]
            img_array = [img_1, img_2, img_3, img_4, img_5, img_6]
            for index, image in enumerate(img_array):
                # dividing by 255 leads to faster convergence through normalization.
                X_loaded_arr[index].append(np.array(image)/(255))
        cv2.imwrite('sample1.jpg', X_loaded_arr[0][0] * 255)
        cv2.imwrite('sample2.jpg', X_loaded_arr[1][0] * 255)
        cv2.imwrite('sample3.jpg', X_loaded_arr[2][0] * 255)
        cv2.imwrite('sample4.jpg', X_loaded_arr[3][0] * 255)
        cv2.imwrite('sample5.jpg', X_loaded_arr[4][0] * 255)
        cv2.imwrite('sample6.jpg', X_loaded_arr[5][0] * 255)

        return [X1_loaded, X2_loaded, X3_loaded, X4_loaded, X5_loaded, X6_loaded]

def train():
    X, y = get_dataset(PATH_TO_DATA)

    #80% of the data will be in our training set
    X_train = np.asarray(left_ult_charge_load_images_for_model(X[0:int(len(X) * 0.8)], train = True))
    y_train = np.asarray(y[0:int(len(X) * 0.8)])

    #20% of the data will be in our validation set
    X_val = np.asarray(left_ult_charge_load_images_for_model(X[int(len(X) * 0.8):], train = True))
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
    model.save('left_ult_model.h5')
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
