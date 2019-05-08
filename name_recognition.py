from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract as py
import argparse
import cv2
import re
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter

py.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
IMAGE_PATH = "name_images/"

#psm 13 or psm8 seems best

def clean_string(names):
    cleaned = ''
    names = ''.join(e for e in names if e.isalnum() or e.isspace())
    for i in re.sub(r'\s+', ' ', names).split():
        if len(i) == 1 or i.islower() or i.lower() == 'ff':
            continue
        cleaned = cleaned + i.upper() + " "

    numNames = len(cleaned.split())
    if numNames == 6:
        msg = "Success ==> Found " + str(numNames) + " ==> "
    else:
        msg = "Fail ==> Found " + str(numNames) + " ==> "

    print(msg + cleaned)
    return cleaned

def killfeed_name_recognition(img_path):
    name_config = ('-l eng --oem 3 --psm 7')
    left,  right = 950, 1280
    upper, lower = 107, 145
    lower_yellow, upper_yellow  = np.array([20, 100, 100]), np.array([40, 255, 255])
    lower_blue, upper_blue = np.array([70, 100, 100]), np.array([130, 255, 255])

    img = cv2.imread(img_path)
    img = cv2.bilateralFilter(img,9,75,75)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    res = cv2.bitwise_and(img, img, mask = mask)

    h, s, v1 = cv2.split(res)

    killfeed_row = v1[upper: lower, 1200: right]
    killfeed_names = py.image_to_string(killfeed_row, config=name_config)
    cv2.imshow('hello', killfeed_row)
    cv2.waitKey(0)

def row_name_recognition(img_path):
    #Variables
    config = ('-l eng --oem 3 --psm 8')
    #name_config = ('-l eng --oem 3 --psm 6')
    left, right = 30, 448
    left1, right1 = 835, 1250
    upper, lower = 75, 90
    left2, right2 = 0, 70

    #Read image and apply bilateral filter
    img = cv2.imread(img_path)
    img = cv2.bilateralFilter(img,9,75,75)
    #Turn image gray, apply Gaussian threshold, Invert Colors
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 2)
    img = cv2.bitwise_not(img)

    #Crop name section
    left_side = img[upper:lower, left:right]
    right_side = img[upper:lower, left1:right1]

    #Convert to Text
    left_names = py.image_to_string(left_side, config=config)
    left_names = clean_string(left_names)
    right_names = py.image_to_string(right_side, config=config)
    right_names = clean_string(right_names)
    return left_names, right_names

    """
    #Get Individual names
    for i in range(0,12):
        if i == 6:
            left2, right2 = 0, 70
        if i < 6:
            name = left_side[0:15, left2:right2]
        else:
            name = right_side[0:15, left2:right2]
        text = py.image_to_string(name, config=name_config)
        text = ''.join(e for e in text if e.isalnum())
        #print("Name: " + text)
        left2 += 70
        right2 += 70
    """

def make_header(full_path):
    print("On Image: " + full_path)
    print("==================================")

if __name__ == '__main__':

    for img in os.listdir(IMAGE_PATH):
        full_path = IMAGE_PATH + img
        make_header(img)
        #killfeed_name_recognition(full_path)
        row_name_recognition(full_path)
        print("")
