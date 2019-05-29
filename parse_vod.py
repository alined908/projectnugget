import cv2
import os
import random
import numpy as np
import imutils

PATH_TO_CLIPS = "vods/"

def parse_video(file_name, full_path, original_count, label):
    count = 0

    video = cv2.VideoCapture(full_path)
    success, frame = video.read()

    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Loading %s (%s), with FPS %d and total frame count %d ' % (file_name, success, fps, total_frame_count))
    while success:
        success, frame = video.read()
        count += 1
        if not success:
            break

        if not count % 15 == 0:
            continue

        if count % 300 == 0:
            print('Currently at frame ', count)
        if np.size(frame, 0) != 720:
            frame = imutils.resize(frame, width=1280)

        cv2.imwrite("vod_data/" + label + '/' + str(original_count * 15 + count) + '.jpg', frame)

    video.release()

def convert_clips():
    for clip_name in os.listdir(PATH_TO_CLIPS):
        if ".mp4" not in clip_name:
            continue
        label = clip_name.split(".mp4")[0]
        os.mkdir("vod_data/" + label)
        parse_video(clip_name, PATH_TO_CLIPS + clip_name, len(os.listdir("vod_data/" + label)), label)
        print("Conversion of " + clip_name + " complete!")

if __name__ == '__main__':
    convert_clips()
