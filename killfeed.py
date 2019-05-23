
import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt


IMAGE_PATH = "E:/Project Nugget/killfeed_images/"

#Killfeed Top Row Coordinates used as baseline
my_y = 107
my_yh = 145
my_x = 950
my_xw = 1280

def get_color_boxes(image, lower_val_1, upper_val_1, lower_val_2, upper_val_2):
    img = image
    hsv1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv1, lower_val_1, upper_val_1)
    _, contours1, _ = cv2.findContours(mask1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    hsv2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv2, lower_val_2, upper_val_2)
    _, contours2, _ = cv2.findContours(mask2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    x1, y1, w1, h1, = -1, -1, -1, -1
    x2, y2, w2, h2, = -1, -1, -1, -1

    color_1_contour_sizes = []
    color_2_contour_sizes = []
    for i, contour in enumerate(contours1):
        contour_area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        color_1_contour_sizes.append([contour_area, x, y, w, h])
    for i, contour in enumerate(contours2):
        contour_area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        color_2_contour_sizes.append([contour_area, x, y, w, h])
    color_1_contour_sizes.sort(reverse = True)
    color_2_contour_sizes.sort(reverse = True)

    yellow_right = True
    b,g,r = img[8, 308]
    if b > r:
      yellow_right = False

    #Do this if yellow/lower_val_1 is on the right
    if yellow_right:
        for contour2 in color_2_contour_sizes:
            for contour1 in color_1_contour_sizes:
                if contour2[1] + contour2[3] < contour1[1]:
                    x1, y1, w1, h1 = contour1[1], contour1[2], contour1[3], contour1[4]
                    x2, y2, w2, h2 = contour2[1], contour2[2], contour2[3], contour2[4]
                    return x1, y1, w1, h1, x2, y2, w2, h2
    else:
    #Do this if blue is on the right
        for contour1 in color_1_contour_sizes:
            for contour2 in color_2_contour_sizes:
                if contour1[1] + contour1[3] < contour2[1]:
                    x1, y1, w1, h1 = contour1[1], contour1[2], contour1[3], contour1[4]
                    x2, y2, w2, h2 = contour2[1], contour2[2], contour2[3], contour2[4]
                    return x1, y1, w1, h1, x2, y2, w2, h2
    return x1, y1, w1, h1, x2, y2, w2, h2

def get_windows(img):
  lower_val, upper_val  = np.array([20, 100, 100]), np.array([40, 255, 255])
  img = cv2.imread(img)[107:350, 950:1280]
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  mask = cv2.inRange(hsv, lower_val, upper_val)
  _, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

  ranges = []

  #Find all possible contours
  for i, contour in enumerate(contours):
      contour_area = cv2.contourArea(contour)
      x,y,w,h = cv2.boundingRect(contour)
      #Only append contours in which the height is greater than 30 but not greater than 40
      #Avoids assist boxes (killing mech, immortality field, bongo) and poorly drawn boxes that overlap
      if (h > 20 and h < 40):
        ranges.append([y,y+h])

  ranges.sort()

  #Keep only the contours that are the outermost (exclude names, character icons)
  #Want to grab the windows for each kill
  unique_ranges = []
  for box in ranges:
    if len(unique_ranges) == 0:
      unique_ranges.append(box)
    else:
      if (box[0] > unique_ranges[-1][1]):
        unique_ranges.append(box)

  #sanity check and draw
  for y in unique_ranges:
    cv2.rectangle(img, (0,y[0]), (330, y[1]), (0, 255, 0), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)
  #plt.imshow(img)

  #Each window should be 38 pixels tall (145-107), with 4 pixels before the top border, and the remaining y pixels allocated below
  y_windows = []
  for y in unique_ranges:
    start = max(y[0] - 4, 0)
    end = start + 38
    y_windows.append([start, end])
  #print(y_windows)
  if len(y_windows) == 0:
      y_windows.append([0,38])
  return(y_windows)

def read_killfeed(image, train, window_y, window_yh):
  img = image
  img = img[my_y + window_y:my_y + window_yh, my_x:my_xw]
  kill_w, kill_h= 36, 26
  assist_w, assist_h  = 16, 22
  hero_coord, kill_coord, death_coord  = [], [], []
  assist_index, assist_coord, contour_coord, bad_contours = [], [], [], []
  lower_yellow, upper_yellow  = np.array([20, 100, 100]), np.array([40, 255, 255])
  lower_blue, upper_blue = np.array([70, 100, 100]), np.array([130, 255, 255])

  #Get the outlining boxes of blue and yellow for each kill
  yellow_x, yellow_y, yellow_w, yellow_h, blue_x, blue_y, blue_w, blue_h = get_color_boxes(img, lower_yellow, upper_yellow, lower_blue, upper_blue)
  if ((yellow_h < kill_h) or (blue_h < kill_h)) and not train:
    return ([[[0, kill_w],[0,kill_h], 'no kill']], [[[0, kill_w],[0, kill_h], 'no kill']], [[[0, assist_w],[0, assist_h], 'no kill']])

  #Get the outlines for the heroes and assists
  imgray = ~cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ret,thresh = cv2.threshold(imgray,170,255,0)
  _, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

  #Find the bounding rectangles and only select the ones that the first index is < 15
  rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  cv2.drawContours(rgb_img, contours, -1, (0, 0, 255), 2)
  cv2.rectangle(rgb_img, (yellow_x, yellow_y), (yellow_x+yellow_w, yellow_y+yellow_h), (0, 255, 0))
  cv2.rectangle(rgb_img, (blue_x, blue_y), (blue_x+blue_w, blue_y+blue_h), (255, 0, 0))
  cv2.imshow('img', rgb_img)
  cv2.waitKey(0)

  right_w = max(yellow_x + yellow_w, blue_x + blue_w)
  left_x = min(yellow_x, blue_x)
  right_x = max(yellow_x, blue_x)

  for i, contour in enumerate(contours):
    contour_area = cv2.contourArea(contour)
    x,y,w,h = cv2.boundingRect(contour)
    #Get only square boxes
    if y < 11 and y >= 4 and ((x > yellow_x  and x < yellow_x + yellow_w) or (x > blue_x  and x < blue_x + blue_w)):
      #Get color of the contour
      pixely = y+1
      if (x+2) >= my_xw - my_x:
          pixelx = my_xw - my_x - 1
      else:
          pixelx = x+2
      b,g,r = img[pixely, pixelx]
      if r > b:
        color = 'yellow'
      else:
        color = 'blue'
      #Get Everyone
      #Make sure only contours passed in are at most within each boxes start and end areas
      #
      if contour_area > 150 and contour_area < 1000 and x + w < right_w and x > left_x and x < right_x + 10:
        contour_coord.append([[x, x+w], [y, y+h], color])
  rgb_img3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  for contour in contour_coord:
      cv2.rectangle(rgb_img3, (contour[0][0], contour[1][0]), (contour[0][1], contour[1][1]), (0, 0, 255))
  cv2.imshow('img', rgb_img3)
  cv2.waitKey(0)

  #Eliminate duplicate or bad contours
  unique_contour_ranges = contour_coord
  for contour in contour_coord:
    start = contour[0][0]
    for unique_contour in unique_contour_ranges:
      if start > unique_contour[0][0] and start < unique_contour[0][1]:
        bad_contours.append(contour)

  unique_contour_ranges = sorted([x for x in unique_contour_ranges if x not in bad_contours])

  #Sort and assign contours based on where the items are (K, A, A, A, D), you know the first and last one are all hero coordinates
  #Kill always first, Death always last, optional assists
  #Resize bounding rectangles, so they are all the same sizes
  num_contours = len(unique_contour_ranges)
  if num_contours < 1:
    #We know there are no kills here, so we should remove these images when processing to save time
    #TODO
    return ([[[0, kill_w],[0,kill_h], 'no kill']], [[[0, kill_w],[0, kill_h], 'no kill']], [[[0, assist_w],[0, assist_h], 'no kill']])
  #print(contour_coord)

  #Find out if yellow/blue is on left or right to determine where kills end and deaths begin
  #Avoids assists running out of bounds
  if yellow_x < blue_x:
      kills_x_end = yellow_x + yellow_w
      deaths_start = blue_x
  else:
      kills_x_end = blue_x + blue_w
      deaths_start = yellow_x

  contour = 0
  for unique_contour in unique_contour_ranges:
    x = unique_contour[0][0]
    y = unique_contour[1][0]
    color = unique_contour[2]
    if contour == 0 or contour == num_contours - 1:
      hero_coord.append([[x, x+kill_w] , [y, y+kill_h], color])
    else:
        #Only append if it is within large contour bound
        if x < kills_x_end and  x > hero_coord[0][0][1]:
          assist_coord.append([[x, x+assist_w], [y, y+ assist_h], color])
    contour += 1


  if train:
      hero_coord = sorted(hero_coord)
      kill_coord.append(hero_coord[0])
      rgb_img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      for i, kill in enumerate(kill_coord):
        cv2.rectangle(rgb_img2, (kill[0][0], kill[1][0]), (kill[0][1], kill[1][1]), (0, 255, 0), 2)
        cv2.rectangle(rgb_img2, (assist_coord[i][0][0], assist_coord[i][1][0]), (assist_coord[i][0][1], assist_coord[i][1][1]), (0, 255, 0), 2)
      cv2.imshow('img', rgb_img2)
      cv2.waitKey(1)
      #print(kill_coord, death_coord, assist_coord)
      return ([kill_coord, [[0, kill_w],[0, kill_h], 'no kill'], assist_coord])
  #Find who is the killer and who is dead
  min_coord, min_index= float('inf'), float('inf')
  if len(hero_coord) == 1 or (blue_x == -1 or yellow_x == -1):
      death_coord.append(hero_coord[0])
      kill_coord.append([[0, kill_w],[0,kill_h], 'no kill'])
  else:
      hero_coord = sorted(hero_coord)
      kill_coord.append(hero_coord[0])
      death_coord.append(hero_coord[1])
  if len(assist_coord) == 0:
      assist_coord = [[[0, assist_w],[0, assist_h], 'no assist']]
  #print(kill_coord, death_coord, assist_coord)

  #print(assist_coord)
  #Draw what we are detecting
  rgb_img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  for i, kill in enumerate(kill_coord):
    cv2.rectangle(rgb_img2, (kill[0][0], kill[1][0]), (kill[0][1], kill[1][1]), (0, 255, 0), 2)
  for i, death in enumerate(death_coord):
    cv2.rectangle(rgb_img2, (death[0][0], death[1][0]), (death[0][1], death[1][1]), (0, 0, 255), 2)
  for assist in assist_coord:
    cv2.rectangle(rgb_img2, (assist[0][0], assist[1][0]), (assist[0][1], assist[1][1]), (255, 0, 0), 2)
  cv2.imshow('img', rgb_img2)
  cv2.waitKey(0)

  #Shift the returned coordinates to include the offset of the window's start values
  for index, y_val in enumerate(kill_coord[0][1]):
      kill_coord[0][1][index] = y_val + window_y
  for index, y_val in enumerate(death_coord[0][1]):
      death_coord[0][1][index] = y_val + window_y
  for index, y_val in enumerate(assist_coord[0][1]):
      assist_coord[0][1][index] = y_val + window_y
  return (kill_coord, death_coord, assist_coord)


if __name__ == '__main__':
    pictures = os.listdir(IMAGE_PATH)
    for index, i in enumerate(pictures):
        print(i)
        windows = get_windows(IMAGE_PATH + i)
        for window in windows:
            read_killfeed(cv2.imread(IMAGE_PATH + i), False, window[0], window[1])
