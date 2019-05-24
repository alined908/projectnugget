import cv2
import numpy as np
import os
import pandas as pd
import keras
from keras.models import load_model
from read_vod_left_side import get_dataset, read_vod_load_left_for_model
from read_vod_right_side import get_dataset, read_vod_load_right_for_model
from read_vod_left_ult import get_dataset, left_ult_charge_load_images_for_model
from read_vod_right_ult import get_dataset, right_ult_charge_load_images_for_model
from gamestate_read_vod import get_dataset, gamestate_load_images_for_model
from pause_read_vod import get_dataset, pause_load_images_for_model
from killfeed_read_vod import get_dataset, killfeed_load_images_for_model
from name_recognition import row_name_recognition

#Necessary dictionaries
paused_state_dic = {'paused':1 , 'live': 0}
ult_charge_dic = {'uncharged':0 , 'charged': 1}
gamestate_dic = {'lobby':1 , 'ingame': 0,}
map_dic = {'BUSAN': 'koth', 'ILIOS': 'koth', 'LJ': 'koth', 'NEPAL': 'koth', 'OASIS': 'koth',
'NUMBANI': 'non-koth', 'KR': 'non-koth', 'BW': 'non-koth', 'KR': 'non-koth', 'HOLLYWOOD': 'non-koth', 'HOLLY': 'non-koth',
'R66': 'non-koth', 'GIB': 'non-koth', 'RIALTO': 'non-koth', 'DORADO': 'non-koth', 'JUNKERTOWN': 'non-koth',
'HANAMURA': 'non-koth', 'LUNAR': 'non-koth', 'PARIS': 'non-koth', 'VOLSKAYA': 'non-koth', 'ANUBIS': 'non-koth'}
hero_dic = {'soldier': 0, 'genji': 1, 'reaper': 2, 'ana': 3, 'bastion': 4, 'brig': 5, 'doomfist': 6, 'dva': 7, 'hanzo':8,
'junkrat': 9, 'lucio': 10, 'mccree': 11, 'mei': 12, 'mercy': 13, 'moira': 14, 'orisa': 15, 'pharah': 16, 'reinhardt': 17,
'roadhog': 18, 'sombra': 19, 'symmetra': 20, 'torbjorn': 21, 'tracer': 22, 'widow': 23, 'winston': 24, 'zarya': 25,
'zenyatta': 26, 'hammond': 27, 'ashe': 28, 'baptiste' : 29, 'unknownhero': 30}
killfeed_dic = {'soldier': 0, 'genji': 1, 'reaper': 2, 'ana': 3, 'bastion': 4, 'brig': 5, 'doomfist': 6, 'dva': 7, 'hanzo':8,
'junkrat': 9, 'lucio': 10, 'mccree': 11, 'mei': 12, 'mercy': 13, 'moira': 14, 'orisa': 15, 'pharah': 16, 'reinhardt': 17,
'roadhog': 18, 'sombra': 19, 'symmetra': 20, 'torbjorn': 21, 'tracer': 22, 'widow': 23, 'winston': 24, 'zarya': 25,
'zenyatta': 26, 'hammond': 27, 'ashe': 28,'baptiste' : 29}

#Inverse dictionaries
inverse_hero = {v: k for k, v in hero_dic.items()}
inverse_ult = {v: k for k, v in ult_charge_dic.items()}
inverse_gamestate = {v: k for k, v in gamestate_dic.items()}
inverse_pause = {v: k for k, v in paused_state_dic.items()}
inverse_killfeed = {v: k for k, v in killfeed_dic.items()}

#Prediction Dictionaries
#What do we predict for each image based off of our models
all_predictions = {'Image': [], 'GameState':[], 'Paused':[], 'Kills': [], 'Kills_Colors': [], 'Deaths': [], 'Deaths_Colors': [],
'Hero 1': [], 'Ult_Charge 1' :[], 'Hero 2': [], 'Ult_Charge 2' :[], 'Hero 3': [], 'Ult_Charge 3' :[],
'Hero 4': [], 'Ult_Charge 4' :[], 'Hero 5': [], 'Ult_Charge 5' :[], 'Hero 6': [], 'Ult_Charge 6' :[],
'Hero 7': [], 'Ult_Charge 7' :[], 'Hero 8': [], 'Ult_Charge 8' :[], 'Hero 9': [], 'Ult_Charge 9' :[],
'Hero 10': [], 'Ult_Charge 10' :[], 'Hero 11': [], 'Ult_Charge 11' :[], 'Hero 12': [], 'Ult_Charge 12' :[]}

#Stores the previously logged change/row that was put into the CSV
previous_prediction = {'Duration': 0, 'GameState':'', 'Paused':'live', 'Kills': [['', '']], 'Deaths': [['', '']],
'Name 1': '', 'Hero 1': ['unknownhero', -1], 'Ult_Charge 1' :['', -1], 'Name 2': '', 'Hero 2': ['unknownhero', -1], 'Ult_Charge 2' :['', -1],
'Name 3': '', 'Hero 3': ['unknownhero', -1], 'Ult_Charge 3' :['', -1], 'Name 4': '', 'Hero 4': ['unknownhero', -1], 'Ult_Charge 4' :['', -1],
'Name 5': '', 'Hero 5': ['unknownhero', -1], 'Ult_Charge 5' :['', -1], 'Name 6': '', 'Hero 6': ['unknownhero', -1], 'Ult_Charge 6' :['', -1],
'Name 7': '', 'Hero 7': ['unknownhero', -1], 'Ult_Charge 7' :['', -1], 'Name 8': '', 'Hero 8': ['unknownhero', -1], 'Ult_Charge 8' :['', -1],
'Name 9': '', 'Hero 9': ['unknownhero', -1], 'Ult_Charge 9' :['', -1], 'Name 10': '', 'Hero 10': ['unknownhero', -1], 'Ult_Charge 10' :['', -1],
'Name 11': '', 'Hero 11': ['unknownhero', -1], 'Ult_Charge 11' :['', -1], 'Name 12': '', 'Hero 12': ['unknownhero', -1], 'Ult_Charge 12' :['', -1]}

#Holds each row of what is going into the CSV
dict = {'Image': [], 'Duration':[], 'GameState':[], 'Paused':[], 'Kills': [], 'Deaths': [],
'Name 1': [], 'Hero 1': [], 'Accuracy 1':[], 'Ult_Charge 1': [], 'Ult_Accuracy 1': [], 'Name 2': [], 'Hero 2': [], 'Accuracy 2':[], 'Ult_Charge 2': [], 'Ult_Accuracy 2': [],
'Name 3': [], 'Hero 3': [], 'Accuracy 3':[], 'Ult_Charge 3': [], 'Ult_Accuracy 3': [], 'Name 4': [], 'Hero 4': [], 'Accuracy 4':[], 'Ult_Charge 4': [], 'Ult_Accuracy 4': [],
'Name 5': [], 'Hero 5': [], 'Accuracy 5':[], 'Ult_Charge 5': [], 'Ult_Accuracy 5': [], 'Name 6': [], 'Hero 6': [], 'Accuracy 6':[], 'Ult_Charge 6': [],  'Ult_Accuracy 6': [],
'Name 7': [], 'Hero 7': [], 'Accuracy 7':[], 'Ult_Charge 7': [], 'Ult_Accuracy 7': [], 'Name 8': [], 'Hero 8': [], 'Accuracy 8':[], 'Ult_Charge 8': [], 'Ult_Accuracy 8': [],
'Name 9': [], 'Hero 9': [], 'Accuracy 9':[], 'Ult_Charge 9': [], 'Ult_Accuracy 9': [], 'Name 10': [], 'Hero 10': [], 'Accuracy 10':[], 'Ult_Charge 10': [], 'Ult_Accuracy 10': [],
'Name 11': [], 'Hero 11': [], 'Accuracy 11':[], 'Ult_Charge 11': [], 'Ult_Accuracy 11': [], 'Name 12': [], 'Hero 12': [], 'Accuracy 12':[], 'Ult_Charge 12': [], 'Ult_Accuracy 12': []}

#load models
right_hero_model = load_model('right_hero_model.h5')
left_hero_model = load_model('left_hero_model.h5')
left_ult_model = load_model('left_ult_model.h5')
right_ult_model = load_model('right_ult_model.h5')
gamestate_model = load_model('gamestate_model.h5')
pause_model = load_model('pause_model.h5')
killfeed_model = load_model('killfeed_model.h5')

def hit_or_miss(data_path, sorted_img_array):
    #Global vars
    scrim_name = data_path.split("/")[1]
    map = scrim_name.split("+")[4]
    opponent = scrim_name.split("+")[3]
    old_date = scrim_name.split("+")[0]
    new_date = old_date.split(".")[1] + "/" + old_date.split(".")[0] + "/" + old_date.split(".")[2]
    #Predict variable with models, builds out all predictions for each image
    predict(sorted_img_array)
    #Clean and create all_predictions to insert the needed items in the actual dict
    clean_predictions(map)
    return new_date, map, opponent
    #print(dict)
"""
Check if we need to relabel the contents of the folder
All images need to be in order in multiples of 15
@vod_path the folder to look in
"""
def relabel_folder_contents(vod_path):
    array_of_sorted_image_names = []
    list_images = os.listdir(vod_path)
    #Make sure it is in order
    for index, number in enumerate(list_images):
        list_images[index] = int(number.split(".")[0])
    list_images.sort()
    copy = list_images.copy()

    #For the ordered list, check to see if all correct numbers are there
    for index, number in enumerate(list_images):
        if list_images[index] == (index+1)*15:
            continue
        else:
            list_images[index] = (index+1)*15

    for index, image in enumerate(copy):
        dest = vod_path + str(list_images[index]) + ".jpg"
        src = vod_path + str(image) + ".jpg"
        array_of_sorted_image_names.append(dest)
        os.rename(src, dest)

    return array_of_sorted_image_names

"""
Based on the models that we have imported create the all_predictions dictionary
"""
def predict(sorted_img_array):
    temp = sorted_img_array
    images = sorted_img_array
    all_predictions['Image'] = temp

    left_hero_array = read_vod_load_left_for_model(temp, resize_to_720P=True, train = False)
    for index, hero in enumerate(left_hero_array):
        left_hero_array[index] = np.asarray(hero)
        left_hero_array[index] = left_hero_model.predict(left_hero_array[index])
        #Update Image #
        for image_index, image in enumerate(left_hero_array[index]):
            image = image.tolist()
            #SLot in hero prediction and accuracy
            all_predictions['Hero ' + str(index + 1)].append([inverse_hero[image.index(max(image))], max(image)])

    right_hero_array = read_vod_load_right_for_model(temp, resize_to_720P=True, train = False)
    for index, hero in enumerate(right_hero_array):
        right_hero_array[index] = np.asarray(hero)
        right_hero_array[index] = right_hero_model.predict(right_hero_array[index])
        #Update Image #
        for image_index, image in enumerate(right_hero_array[index]):
            image = image.tolist()
            #SLot in hero prediction and accuracy
            all_predictions['Hero ' + str(index + 7)].append([inverse_hero[image.index(max(image))], max(image)])

    left_ult_array = left_ult_charge_load_images_for_model(temp, resize_to_720P=True, train = False)
    for index, ult in enumerate(left_ult_array):
        left_ult_array[index] = np.asarray(ult)
        left_ult_array[index] = left_ult_model.predict(left_ult_array[index])
        for image_index, image in enumerate(left_ult_array[index]):
            #SLot in ult prediction and accuracy
            image = image.tolist()
            all_predictions['Ult_Charge ' + str(index + 1)].append([inverse_ult[image.index(max(image))], max(image)])

    right_ult_array = right_ult_charge_load_images_for_model(temp, resize_to_720P=True, train = False)
    for index, ult in enumerate(right_ult_array):
        right_ult_array[index] = np.asarray(ult)
        right_ult_array[index] = right_ult_model.predict(right_ult_array[index])
        for image_index, image in enumerate(right_ult_array[index]):
            #SLot in ult prediction and accuracy
            image = image.tolist()
            all_predictions['Ult_Charge ' + str(index + 7)].append([inverse_ult[image.index(max(image))], max(image)])

    gamestate_array = gamestate_load_images_for_model(temp, resize_to_720P= True, train=False)
    gamestate_predict = gamestate_model.predict(np.asarray(gamestate_array))
    for index, image in enumerate(gamestate_predict):
        img = image.tolist()
        all_predictions['GameState'].append([inverse_gamestate[img.index(max(img))], max(img)])

    paused_array = pause_load_images_for_model(temp, resize_to_720P = True, train = False)
    paused_predict = pause_model.predict(np.asarray(paused_array))
    for index, image in enumerate(paused_predict):
        img = image.tolist()
        all_predictions['Paused'].append([inverse_pause[img.index(max(img))], max(img)])

    kills_array, deaths_array, assists_array, kills_colors, deaths_colors, assists_colors = killfeed_load_images_for_model(temp, resize_to_720P = True, train = False)
    for element in kills_array:
        img_kills = []
        kills_predict = killfeed_model.predict(np.asarray(element))
        for index, image in enumerate(kills_predict):
            img = image.tolist()
            img_kills.append([inverse_killfeed[img.index(max(img))], max(img)])
        all_predictions['Kills'].append(img_kills)
    for element in deaths_array:
        img_deaths = []
        deaths_predict = killfeed_model.predict(np.asarray(element))
        for index, image in enumerate(deaths_predict):
            img = image.tolist()
            img_deaths.append([inverse_killfeed[img.index(max(img))], max(img)])
        all_predictions['Deaths'].append(img_deaths)
    all_predictions['Kills_Colors'] = kills_colors
    all_predictions['Deaths_Colors'] = deaths_colors

#check for duration difference
def image_difference_check(previmage, currimage):
    prev_num = int(previmage.split("/")[2].split(".")[0])
    curr_num = int(currimage.split("/")[2].split(".")[0])
    if curr_num - prev_num >= 600:
        return True
    else:
        return False

def get_start_frames(map_type, roundcounter, frames_to_start):
    if map_type == "non-koth":
        if roundcounter > 1:
            start_frames = frames_to_start[1]
        else:
            start_frames = frames_to_start[0]
    else:
        if roundcounter > 0:
            start_frames = frames_to_start[1]
        else:
            start_frames = frames_to_start[0]

    return start_frames

def get_name_from_color(kill_color, death_color, kill_hero, death_hero):
    kill_index, death_index = -1, -1
    killer, death = "", ""

    if kill_color == death_color:
        print("Bug: Kill color is same as death color.")
        return 'Not Right', 'Not Right'
    if not (kill_color == 'yellow' or kill_color == 'blue') or not (death_color == 'yellow' or death_color == 'blue'):
        print("Bug: Kill color is not yellow or blue.")
        return 'Not Right', 'Not Right'

    for hero_num in range(1,13):
        hero = dict['Hero ' + str(hero_num)][-1]

        if hero == kill_hero and hero_num < 7:
            if kill_color == 'yellow':
                kill_index = hero_num
        if hero == kill_hero and hero_num >= 7:
            if kill_color == 'blue':
                kill_index = hero_num
        if hero == death_hero and hero_num < 7:
            if death_color == 'yellow':
                death_index = hero_num
        if hero == death_hero and hero_num >= 7:
            if death_color == 'blue':
                death_index = hero_num

    if kill_index == -1 or death_index == -1:
        return 'Not Right', 'Not Right'

    killer = previous_prediction['Name ' + str(kill_index)]
    death = previous_prediction['Name ' + str(death_index)]
    return killer, death

def append_previous():
    for hero_num in range(1,13):
        dict['Hero ' + str(hero_num)].append(previous_prediction['Hero ' + str(hero_num)][0])
        dict['Accuracy ' + str(hero_num)].append(previous_prediction['Hero ' + str(hero_num)][1])
        dict['Ult_Charge ' + str(hero_num)].append(previous_prediction['Ult_Charge ' + str(hero_num)][0])
        dict['Ult_Accuracy ' + str(hero_num)].append(previous_prediction['Ult_Charge ' + str(hero_num)][1])

def get_row_of_names(img_path):
    left_name_str, right_name_str = row_name_recognition(img_path)
    left_names = left_name_str.split(" ")
    right_names = right_name_str.split(" ")
    names = left_names + right_names
    for hero_num in range(1, 13):
        dict['Name ' + str(hero_num)].append(names[hero_num - 1])
        previous_prediction['Name ' + str(hero_num)] = names[hero_num - 1]

def clean_predictions(map):
    pause_flag, roundflag, initflag, nameflag = False, True, True, False
    roundstart, roundcounter = 0, 0
    HERO_THRESHOLD, ULT_THRESHOLD = 0.95, 0.90
    LOBBY_THRESHOLD, PAUSE_THRESHOLD = 0.90, 0.90
    KD_THRESHOLD, DURATION_THRESHOLD  = 0.985, 9

    #Frames different between load in and actual round start
    map_type = map_dic[map]
    if map_type == 'koth':
        frames_to_start = [4335, 1830]
    else:
        frames_to_start = [5100, 4230]

    #For each image, go through and see if an important change happens. If yes, update image.
    for i in range(len(all_predictions['Image'])):
        if (i % 100) == 0: print("On Image: ", i)
        pause_flag = False
        update_image = False
        changed = set()
        all_heroes = {1,2,3,4,5,6,7,8,9,10,11,12}
        #Check to see if in lobby state, if so set heroes and ults to default values
        #If previous predictions were lobby then skip to next image
        if ((all_predictions['GameState'][i][0] == 'lobby') and (all_predictions['GameState'][i][1] > LOBBY_THRESHOLD)):
            if previous_prediction['GameState'] != all_predictions['GameState'][i][0]:
                dict['GameState'].append(all_predictions['GameState'][i][0])
                dict['Image'].append(all_predictions['Image'][i])
                if roundstart < 1:
                    dict['Duration'].append(0)
                else:
                    dict['Duration'].append(previous_prediction['Duration'])
                dict['Paused'].append('live')
                for hero_num in range(1,13):
                    dict['Hero ' + str(hero_num)].append('')
                    dict['Accuracy ' + str(hero_num)].append('')
                    dict['Ult_Charge ' + str(hero_num)].append('')
                    dict['Ult_Accuracy ' + str(hero_num)].append('')
                    dict['Name ' + str(hero_num)].append('')
                dict['Kills'].append(['', ''])
                dict['Deaths'].append(['', ''])
            previous_prediction['GameState'] = all_predictions['GameState'][i][0]
            previous_prediction['Paused'] = 'live'
            previous_prediction['Duration'] = dict['Duration'][-1]
            continue

        #Moved from lobby state to ingame, update image and look for heroes
        if ((previous_prediction['GameState'] != 'ingame') and (all_predictions['GameState'][i][0] == 'ingame') and (all_predictions['GameState'][i][1] > LOBBY_THRESHOLD)):
            #print("A")
            dict['GameState'].append(all_predictions['GameState'][i][0])
            previous_prediction['GameState'] = all_predictions['GameState'][i][0]
            dict['Image'].append(all_predictions['Image'][i])
            update_image = True

        #Moved from live to paused
        if ((all_predictions['Paused'][i][0] == 'paused') and (all_predictions['Paused'][i][1] > PAUSE_THRESHOLD)):
            #print("B")
            dict['Paused'].append(all_predictions['Paused'][i][0])
            previous_prediction['Paused'] = all_predictions['Paused'][i][0]
            if not update_image:
                dict['Image'].append(all_predictions['Image'][i])
                update_image = True
            pause_flag = True

        if pause_flag:
            append_previous()
        else:
            #Predict for each hero and ult charge what they will be
            for hero_num in range(1,13):
                if ((previous_prediction['Hero ' + str(hero_num)][0] != all_predictions['Hero ' + str(hero_num)][i][0] and all_predictions['Hero ' + str(hero_num)][i][1] > HERO_THRESHOLD)
                or (previous_prediction['Ult_Charge ' + str(hero_num)][0] != all_predictions['Ult_Charge ' + str(hero_num)][i][0] and all_predictions['Ult_Charge ' + str(hero_num)][i][1] > ULT_THRESHOLD)):
                    #print("C")
                    #Add image
                    if not update_image:
                        dict['Image'].append(all_predictions['Image'][i])
                        update_image = True
                    if(((previous_prediction['Hero ' + str(hero_num)][0] != all_predictions['Hero ' + str(hero_num)][i][0]) and (all_predictions['Hero ' + str(hero_num)][i][1] > HERO_THRESHOLD))
                    and ((previous_prediction['Ult_Charge ' + str(hero_num)][0] != all_predictions['Ult_Charge ' + str(hero_num)][i][0]) and (all_predictions['Ult_Charge ' + str(hero_num)][i][1] > ULT_THRESHOLD))):
                        #Update Hero, Accuracy, and ult charge
                        new_hero = all_predictions['Hero ' + str(hero_num)][i][0]
                        new_Ult_Charge = all_predictions['Ult_Charge ' + str(hero_num)][i][0]
                        dict['Hero ' + str(hero_num)].append(new_hero)
                        dict['Accuracy ' + str(hero_num)].append(all_predictions['Hero ' + str(hero_num)][i][1])
                        if new_hero == "unknowhero":
                            dict['Ult_Charge ' + str(hero_num)].append('uncharged')
                            previous_prediction['Ult_Charge ' + str(hero_num)][0] = 'uncharged'
                        else:
                            dict['Ult_Charge ' + str(hero_num)].append(new_Ult_Charge)
                            previous_prediction['Ult_Charge ' + str(hero_num)][0] = new_Ult_Charge
                        dict['Ult_Accuracy ' + str(hero_num)].append(all_predictions['Ult_Charge ' + str(hero_num)][i][1])
                        previous_prediction['Ult_Charge ' + str(hero_num)][0] = new_Ult_Charge
                        previous_prediction['Hero ' + str(hero_num)][0] = new_hero
                        previous_prediction['Hero ' + str(hero_num)][1] = all_predictions['Hero ' + str(hero_num)][i][1]
                        previous_prediction['Ult_Charge ' + str(hero_num)][1] = all_predictions['Ult_Charge ' + str(hero_num)][i][1]
                    elif (previous_prediction['Hero ' + str(hero_num)][0] != all_predictions['Hero ' + str(hero_num)][i][0] and all_predictions['Hero ' + str(hero_num)][i][1] > HERO_THRESHOLD):
                        #Update Hero, Accuracy, reuse ult charge
                        new_hero = all_predictions['Hero ' + str(hero_num)][i][0]
                        dict['Hero ' + str(hero_num)].append(new_hero)
                        dict['Accuracy ' + str(hero_num)].append(all_predictions['Hero ' + str(hero_num)][i][1])
                        dict['Ult_Charge ' + str(hero_num)].append(previous_prediction['Ult_Charge ' + str(hero_num)][0])
                        #AA
                        dict['Ult_Accuracy ' + str(hero_num)].append(previous_prediction['Ult_Charge ' + str(hero_num)][1])
                        previous_prediction['Hero ' + str(hero_num)][0] = new_hero
                        previous_prediction['Hero ' + str(hero_num)][1] = all_predictions['Hero ' + str(hero_num)][i][1]
                    elif (previous_prediction['Ult_Charge ' + str(hero_num)][0] != all_predictions['Ult_Charge ' + str(hero_num)][i][0] and all_predictions['Ult_Charge ' + str(hero_num)][i][1] > ULT_THRESHOLD):
                        #Update ult charge, reuse Hero and accuracy
                        new_Ult_Charge = all_predictions['Ult_Charge ' + str(hero_num)][i][0]
                        dict['Ult_Charge ' + str(hero_num)].append(new_Ult_Charge)
                        dict['Hero ' + str(hero_num)].append(previous_prediction['Hero ' + str(hero_num)][0])
                        dict['Accuracy ' + str(hero_num)].append(previous_prediction['Hero ' + str(hero_num)][1])
                        dict['Ult_Accuracy ' + str(hero_num)].append(all_predictions['Ult_Charge ' + str(hero_num)][i][1])
                        previous_prediction['Ult_Charge ' + str(hero_num)][0] = new_Ult_Charge
                        previous_prediction['Ult_Charge ' + str(hero_num)][1] = all_predictions['Ult_Charge ' + str(hero_num)][i][1]
                    changed.add(hero_num)

        if update_image and not pause_flag:
            #Repeat previous hero information for people who did not switch or charge ult (unchanged heroes/columns)
            all_heroes.difference_update(changed)
            for hero_num in all_heroes:
                 dict['Hero ' + str(hero_num)].append(previous_prediction['Hero ' + str(hero_num)][0])
                 dict['Accuracy ' + str(hero_num)].append(previous_prediction['Hero ' + str(hero_num)][1])
                 dict['Ult_Charge ' + str(hero_num)].append(previous_prediction['Ult_Charge ' + str(hero_num)][0])
                 dict['Ult_Accuracy ' + str(hero_num)].append(previous_prediction['Ult_Charge ' + str(hero_num)][1])

        if len(dict['GameState']) > len(dict['Hero 1']) or (len(dict['Paused']) > len(dict['Hero 1'])):
            for hero_num in range(1,13):
                 dict['Hero ' + str(hero_num)].append("")
                 dict['Accuracy ' + str(hero_num)].append("")
                 dict['Ult_Charge ' + str(hero_num)].append("")
                 dict['Ult_Accuracy ' + str(hero_num)].append("")
        if len(dict['GameState']) < len(dict['Hero 1']):
            dict['GameState'].append(previous_prediction['GameState'])
        if len(dict['Paused']) < len(dict['Hero 1']):
            dict['Paused'].append(previous_prediction['Paused'])

        #Get start frames
        start_frames = get_start_frames(map_type, roundcounter, frames_to_start)

        #Add a row to know at what image we start counting duration from 0 aka real round start
        if (((i+1)*15) == (roundstart + start_frames)):
            #Append info for time if new row is inserted
            roundcounter += 1
            if not update_image:
                #print("G")
                dict['Duration'].append(dict['Duration'][-1])
                dict['GameState'].append(all_predictions['GameState'][i-1][0])
                dict['Image'].append(all_predictions['Image'][i-1])
                dict['Paused'].append('live')
                append_previous()

        #Killfeed Stuff
        if len(all_predictions['Kills'][i]) != len(all_predictions['Deaths'][i]):
            print("Differing # of kills and deaths events")

        temp_prev_kill, temp_prev_death = 'unchanged', 'unchanged'
        loop_counter = 0
        for index, event in enumerate(all_predictions['Kills'][i]):
            curr_kill, curr_kill_acc, curr_kill_color = event[0], event[1], all_predictions['Kills_Colors'][i][index]
            curr_death, curr_death_acc, curr_death_color = all_predictions['Deaths'][i][index][0], all_predictions['Deaths'][i][index][1], all_predictions['Deaths_Colors'][i][index]
            prev_kill, prev_kill_color, prev_death, prev_death_color = previous_prediction['Kills'][0][0], previous_prediction['Kills'][0][1], previous_prediction['Deaths'][0][0], previous_prediction['Deaths'][0][1]
            good_flag = False

            if (curr_kill_acc > KD_THRESHOLD) and (curr_death_acc > KD_THRESHOLD) and (curr_kill_color != 'no kill') and (curr_death_color != 'no kill'):
                if (curr_kill != prev_kill) and (curr_death != prev_death):
                    good_flag = True
                else:
                    if image_difference_check(dict['Image'][-1], all_predictions['Image'][i]):
                        good_flag = True
                    else:
                        break

            if good_flag:
                kill_player, death_player = get_name_from_color(curr_kill_color, curr_death_color, curr_kill, curr_death)
                if not kill_player == 'Not Right':
                    loop_counter += 1
                    dict['Kills'].append([kill_player, curr_kill])
                    dict['Deaths'].append([death_player, curr_death])
                    if loop_counter == 1:
                        temp_prev_kill, temp_prev_kill_color = dict['Kills'][-1][1], curr_kill_color
                        temp_prev_death, temp_prev_death_color = dict['Deaths'][-1][1], curr_death_color
                    if len(dict['Image']) < len(dict['Kills']):
                        dict['Image'].append(all_predictions['Image'][i])
                    update_image = True

                    #kill/death but no change
                    dict['GameState'].append(all_predictions['GameState'][i-1][0])
                    dict['Paused'].append('live')
                    for hero_num in range(1, 13):
                        if len(dict['Hero ' + str(hero_num)]) < len(dict['Image']):
                             dict['Hero ' + str(hero_num)].append(previous_prediction['Hero ' + str(hero_num)][0])
                        if len(dict['Accuracy ' + str(hero_num)]) < len(dict['Image']):
                            dict['Accuracy ' + str(hero_num)].append(previous_prediction['Hero ' + str(hero_num)][1])
                        if len(dict['Ult_Charge ' + str(hero_num)]) < len(dict['Image']):
                            dict['Ult_Charge ' + str(hero_num)].append(previous_prediction['Ult_Charge ' + str(hero_num)][0])
                        if len(dict['Ult_Accuracy ' + str(hero_num)]) < len(dict['Image']):
                            dict['Ult_Accuracy ' + str(hero_num)].append(previous_prediction['Ult_Charge ' + str(hero_num)][1])
                        dict['Name ' + str(hero_num)].append(previous_prediction['Name ' + str(hero_num)])

        if temp_prev_kill != 'unchanged' and temp_prev_kill != 'unchanged':
            previous_prediction['Kills'][0][0], previous_prediction['Kills'][0][1] = temp_prev_kill, temp_prev_kill_color
            previous_prediction['Deaths'][0][0], previous_prediction['Deaths'][0][1] = temp_prev_death, temp_prev_death_color
        #No kill/death but hero swap or ult charge
        if len(dict['Kills']) < len(dict['Hero 1']):
            dict['Kills'].append(['', ''])
            dict['Deaths'].append(['', ''])

        #Handle Duration
        if update_image:
            #Lobby Status
            currImage = int(dict['Image'][-1].split("/")[2].split('.')[0])
            #Paused Status
            if dict['Paused'][-1] == 'paused':
                dict['Duration'].append(dict['Duration'][-1])
                roundstart = currImage
            #Ingame and playing
            else:
                #Check for start of rounds
                check = []
                for hero_num in range(1,13):
                    check.append(dict['Hero ' + str(hero_num)][-1])
                occurrences = check.count("unknownhero")
                #This row is a new round
                if ((i+1)*15) >= (roundstart + start_frames):
                    roundflag = True
                if occurrences >= DURATION_THRESHOLD and dict['GameState'][-1] == 'ingame':
                    if roundflag:
                        dict['Duration'].append(previous_prediction['Duration'])
                        if (roundcounter == 0 and initflag) or ((i+1)*15) >= (roundstart + start_frames):
                            if not nameflag:
                                get_row_of_names(all_predictions['Image'][i])
                                nameflag = True
                            roundstart = currImage
                            initflag = False
                        if roundcounter != 0:
                            roundflag = False
                    else:
                        if currImage < roundstart + start_frames:
                            dict['Duration'].append(dict['Duration'][-1])
                        else:
                            prevImage = int(dict['Image'][-2].split("/")[2].split('.')[0])
                            timediff = (currImage - prevImage)/60
                            dict['Duration'].append(timediff  + dict['Duration'][-1])
                #Still in same round
                else:
                    #2040 is how many frames it takes between loading in and the start of round for non-koth
                    if dict['Paused'][-2] == 'paused' or currImage < roundstart + start_frames:
                        dict['Duration'].append(dict['Duration'][-1])
                    else:
                        prevImage = int(dict['Image'][-2].split("/")[2].split('.')[0])
                        timediff = (currImage - prevImage)/60
                        dict['Duration'].append(timediff  + dict['Duration'][-1])
                previous_prediction['Duration'] = dict['Duration'][-1]

        if len(dict['Name 1']) < len(dict['Hero 1']):
            for hero_num in range(1, 13):
                dict['Name ' + str(hero_num)].append(previous_prediction['Name ' + str(hero_num)])

        if len(dict['Duration']) < len(dict['Kills']):
            difference = len(dict['Kills']) - len(dict['Duration'])
            for loop in range(difference):
                dict['Duration'].append(previous_prediction['Duration'])

        if len(dict['Image']) < len(dict['Kills']):
            dict['Image'].append(all_predictions['Image'][i])

    for key in dict.keys():
        print(key + ": " + str(len(dict[key])))

"""
Make the beautiful CSV that we said we would
"""
if __name__ == '__main__':

    csv_folder = "csvs/to_csv/"
    scrim_folder = "vod_data/"
    for folder in os.listdir(scrim_folder):
        vod_path = scrim_folder + folder + "/"
        print("================================================")
        print("Relabeling files")
        print("================================================")
        sorted_image_array = relabel_folder_contents(vod_path)
        print("Currently processing:", folder)
        print("===============================================")
        date, map, opponent = hit_or_miss(vod_path, sorted_image_array)
        df = pd.DataFrame(data = dict)
        df.insert(1, 'Map', map)
        df.insert(2, 'Opponent', opponent)
        df.insert(1, 'Date', date)
        df.to_csv(csv_folder + folder + "+kdtest3.csv", sep=',')
        print("Successfully created csv!")
    """
    csv_folder = "csvs/"
    scrim_folder = "vod_data/"
    for folder in os.listdir(scrim_folder):
        vod_path = scrim_folder + folder + "/"
        print("================================================")
        print("Relabeling files")
        print("Relabeling: ", folder)
        print("================================================")
    vod_path = "vod_data/01.04.2019+SF+vs+DAL+RIALTO/"
    sorted_img_array = relabel_folder_contents(vod_path)
    print("================================================")
    print("Loading Images")
    print("================================================")
    temp = sorted_img_array
    images = sorted_img_array

    kills_array, deaths_array, assists_array, kills_colors, deaths_colors, assists_colors = killfeed_load_images_for_model(temp, resize_to_720P = True, train = False)
    print("================================================")
    print("Predicting Images")
    print("================================================")
    all_predictions = {'Images':[], 'Kills':[], 'Deaths': []}

    for element in kills_array:
        img_kills = []
        kills_predict = killfeed_model.predict(np.asarray(element))
        for index, image in enumerate(kills_predict):
            img = image.tolist()
            img_kills.append([inverse_killfeed[img.index(max(img))], max(img)])
        all_predictions['Kills'].append(img_kills)
    for element in deaths_array:
        img_deaths = []
        deaths_predict = killfeed_model.predict(np.asarray(element))
        for index, image in enumerate(deaths_predict):
            img = image.tolist()
            img_deaths.append([inverse_killfeed[img.index(max(img))], max(img)])
        all_predictions['Deaths'].append(img_deaths)
    all_predictions['Images'] = temp
    all_predictions['Kills_Colors'] = kills_colors
    all_predictions['Deaths_Colors'] = deaths_colors

    print("================================================")
    print("Filtering Predictions")
    print("================================================")
    #print(all_predictions)
    for key in all_predictions.keys():
        print(key + ": " + str(len(all_predictions[key])))
    df = pd.DataFrame(data = all_predictions)
    df.to_csv(csv_folder + "STUPID+multitest.csv", sep=',')
    """
