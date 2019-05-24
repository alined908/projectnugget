import cv2
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from killfeed import read_killfeed, get_windows
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Input, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from random import shuffle

PATH_TO_DATA = 'data/killfeed_data/killdeath_data/'
PATH_TO_ASSISTS = 'data/killfeed_data/assist_data/'

label_dic = {'soldier': 0, 'genji': 1, 'reaper': 2, 'ana': 3, 'bastion': 4, 'brig': 5, 'doomfist': 6, 'dva': 7, 'hanzo':8,
'junkrat': 9, 'lucio': 10, 'mccree': 11, 'mei': 12, 'mercy': 13, 'moira': 14, 'orisa': 15, 'pharah': 16, 'reinhardt': 17,
'roadhog': 18, 'sombra': 19, 'symmetra': 20, 'torbjorn': 21, 'tracer': 22, 'widow': 23, 'winston': 24, 'zarya': 25,
'zenyatta': 26, 'hammond': 27, 'ashe': 28,'baptiste' : 29}

assist_dic = {'ana': 0, 'brig': 1, 'junkrat': 2, 'lucio': 3, 'mccree': 4, 'mei': 5, 'mercy': 6,  'orisa': 7,
'reinhardt': 8, 'roadhog': 9, 'sombra': 10,   'zarya': 11,'zenyatta': 12, 'ashe': 13,'baptiste' : 14}

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
    if get_test_set:
        zipped = list(zip(X, y))
        shuffle(zipped)
        X, y = zip(*zipped)
    return X, y

def get_model(kill):
    if kill:
        size = 30
        first = 26
        second = 36
    else:
        size = 15
        first = 22
        second = 16

    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape=(first, second, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    if kill:
        model.add(Conv2D(32, (3,3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(size))
    model.add(Activation('softmax'))
    opt = keras.optimizers.Adam(lr=0.0001)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.summary()
    return model

def killfeed_load_images_for_model(X_batch, resize_to_720P=True, train=False):
    # X_batch is just a bunch of file names. We need to load the image to pass it to a net!
    X_loaded_kills = []
    X_loaded_deaths = []
    X_loaded_assists = []
    kill_colors = []
    death_colors = []
    assists_colors = []
    if train:
        for index, path in enumerate(X_batch):
            #print("Image is: ", path)
            img = cv2.imread(path)
            kill_coord, death_coord, assist_coord = read_killfeed(img, train, 0, 32)
            # Resize image
            x, xw, y, yh = kill_coord[0][0][0], kill_coord[0][0][1], kill_coord[0][1][0], kill_coord[0][1][1]
            #print("Assist coord", assist_coord)
            ax, axw, ay, ayh = assist_coord[0][0][0], assist_coord[0][0][1], assist_coord[0][1][0], assist_coord[0][1][1]
            assist_img = img[109 + ay: 109 + ayh, 950+ax:950+axw]
            img = img[109 + y:109 + yh, 950+x  :950+xw]
            # dividing by 255 leads to faster convergence through normalization.
            X_loaded_kills.append(np.array(img)/(255))
            X_loaded_assists.append(np.array(assist_img)/(255))
            if index < 3:
                print("X_loaded_kills: ", X_loaded_kills)
                print("X_loaded_assists: ", X_loaded_assists)

        cv2.imwrite('kill_sample.jpg', X_loaded_kills[0] * 255)
        cv2.imwrite('assist_sample.jpg', X_loaded_assists[0] * 255)
        return X_loaded_kills, X_loaded_assists
    else:
        for path in X_batch:
            num_loaded = len(X_loaded_kills)
            if num_loaded % 100 == 0:
                print("Killfeed: Processed " + str(num_loaded)+ " images already")
            img = cv2.imread(path)
            windows = get_windows(path)
            frame_kills = []
            frame_deaths = []
            frame_assists = []
            frame_kill_colors = []
            frame_death_colors = []
            frame_assists_colors = []
            for window in windows:
                kill_coord, death_coord, assist_coord = read_killfeed(cv2.imread(path), False, window[0], window[1])
                x, xw, y, yh = kill_coord[0][0][0], kill_coord[0][0][1], kill_coord[0][1][0], kill_coord[0][1][1]
                img1 = img[109 + y:109 + yh, 950+x  :950+xw]
                frame_kills.append(np.array(img1)/(255))
                frame_kill_colors.append(kill_coord[0][2])

                x, xw, y, yh = death_coord[0][0][0], death_coord[0][0][1], death_coord[0][1][0], death_coord[0][1][1]
                img2 = img[109 + y:109 + yh, 950+x  :950+xw]
                shape = (np.array(img2)/255).shape
                if shape != (26,36,3):
                    print(path)
                frame_deaths.append(np.array(img2)/(255))
                frame_death_colors.append(death_coord[0][2])

                for assist in assist_coord:
                    x, xw, y, yh = assist[0][0], assist[0][1], assist[1][0], assist[1][1]
                    img3 = img[109 + y:109 + yh, 950+x  :950+xw]
                    frame_assists.append(np.array(img3)/(255))
                frame_assists_colors.append(assist_coord[0][2])
            X_loaded_kills.append(frame_kills)
            X_loaded_deaths.append(frame_deaths)
            X_loaded_assists.append(frame_assists)
            kill_colors.append(frame_kill_colors)
            death_colors.append(frame_death_colors)
            assists_colors.append(frame_assists_colors)

        #cv2.imwrite('sample.jpg', X_loaded_kills[10][0] * 255)
        return [X_loaded_kills, X_loaded_deaths, X_loaded_assists, kill_colors, death_colors, assists_colors]

def train():
    #X, y = get_dataset(PATH_TO_DATA)
    X2, y2 = get_dataset(PATH_TO_ASSISTS)

    #80% of the data will be in our training set
    #X_train_kills = np.asarray(killfeed_load_images_for_model(X, train = True))
    X_train_assists = np.asarray(killfeed_load_images_for_model(X2, train = True))
    #y_train_kills = np.asarray(y)
    y_train_assists = np.asarray(y2)

    #model = get_model(True)
    #model.compile(loss='categorical_crossentropy',
                 # optimizer = keras.optimizers.Adam(lr=0.0001),
                  #metrics=['accuracy'])

    #model.summary()
    #num_epochs = 130
    #batch_size = 4

    #print("Beginning training!")

    #model_hist = model.fit(x = X_train_kills, y = y_train_kills, batch_size = batch_size, epochs = num_epochs, verbose = 1)
    #model.save('killdeath_model.h5')
    #print("Training complete!")

    #Assist model
    model = get_model(False)
    model.compile(loss='categorical_crossentropy',
                  optimizer = keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy'])

    model.summary()
    num_epochs = 90
    batch_size = 4

    print("Beginning training!")
    print(X_train_assists.shape)

    model_hist = model.fit(x = X_train_assists, y = y_train_assists, batch_size = batch_size, epochs = num_epochs, verbose = 1)
    model.save('assists_model.h5')

    print("Training complete!")

if __name__ == '__main__':
    train()
