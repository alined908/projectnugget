import numpy as np
import cv2
import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

IMAGE_PATH = "E:/Project Nugget/killfeed_images/"

#Killfeed Top Row Coordinates
my_y = 107
my_yh = 145
my_x = 950
my_xw = 1280

def get_color_boxes(image, lower_val, upper_val):
    color_contour_area = 0
    img = image
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_val, upper_val)
    res = cv2.bitwise_and(img, img, mask = mask)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h, = -1, -1, -1, -1
    for i, contour in enumerate(contours):
        contour_area = cv2.contourArea(contour)
        if contour_area > color_contour_area:
            color_contour_area = contour_area
            x, y, w, h = cv2.boundingRect(contour)
    return x, y, w, h

def read_killfeed(image, train):
  img = image
  img = img[my_y:my_yh, my_x:my_xw]
  kill_w, kill_h= 36, 26
  assist_w, assist_h  = 16, 22
  hero_coord, kill_coord, death_coord  = [], [], []
  assist_index, assist_coord, contour_coord, bad_contours = [], [], [], []
  lower_yellow, upper_yellow  = np.array([20, 100, 100]), np.array([40, 255, 255])
  lower_blue, upper_blue = np.array([70, 100, 100]), np.array([130, 255, 255])

  #Get the outlining boxes of blue and yellow for each kill
  yellow_x, yellow_y, yellow_w, yellow_h = get_color_boxes(img, lower_yellow, upper_yellow)
  blue_x, blue_y, blue_w, blue_h = get_color_boxes(img, lower_blue, upper_blue)
  #If the height of these boxes are not at least as big as a kill then that means these are the mini boxes which we will not deal with
  if ((yellow_h < kill_h) or (blue_h < kill_h)) and not train:
      return ([[[0, kill_w],[0,kill_h], 'no kill']], [[[0, kill_w],[0, kill_h], 'no kill']], [[[0, assist_w],[0, assist_h], 'no kill']])

  #Get the outlines for the heroes and assists
  imgray = ~cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ret,thresh = cv2.threshold(imgray,170,255,0)
  _, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

  #Find the bounding rectangles and only select the ones that the first index is < 15
  rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  cv2.drawContours(rgb_img, contours, -1, (0, 0, 255), 2)
  cv2.rectangle(rgb_img, (yellow_x, yellow_y), (yellow_x+yellow_w, yellow_y+yellow_h), (0, 0, 255))
  cv2.rectangle(rgb_img, (blue_x, blue_y), (blue_x+blue_w, blue_y+blue_h), (0, 0, 255))
  #xd = plt.imshow(rgb_img)
  #plt.show()

  for i, contour in enumerate(contours):
    contour_area = cv2.contourArea(contour)
    x,y,w,h = cv2.boundingRect(contour)
    #Get only square boxes
    if y < 11 and y >= 5 and ((x > yellow_x  and x < yellow_x + yellow_w) or (x > blue_x  and x < blue_x + blue_w)):
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
      if contour_area > 40 and contour_area < 1000:
        contour_coord.append([[x, x+w], [y, y+h], color])

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
    #print(x)
    #print(kills_x_end)
    color = unique_contour[2]
    if contour == 0 or contour == num_contours - 1:
      hero_coord.append([[x, x+kill_w] , [y, y+kill_h], color])
    else:
        #Only append if it is within large contour bound
        if x < kills_x_end and  x > hero_coord[0][0][1]:
          assist_coord.append([[x, x+assist_w], [y, y+ assist_h], color])
    contour += 1
  """
  bad_assists = []
  for hero in hero_coord:
      for index, assist in enumerate(assist_coord):
          if hero[0][0] < assist[0][0] < hero[0][1]:
            bad_assists.append(index)

  for bad in bad_assists:
      del assist_coord[bad]
  """
  #print("hero coord ", hero_coord)

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

  #cv2.imshow('img', rgb_img2)
  #cv2.waitKey(1)
  #plt.show()
  return (kill_coord, death_coord, assist_coord)


if __name__ == '__main__':
    pictures = os.listdir(IMAGE_PATH)
    for index, i in enumerate(pictures):
        print(i)
        read_killfeed(cv2.imread(IMAGE_PATH + i))
