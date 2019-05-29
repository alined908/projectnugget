import cv2
import os
import random
import numpy as np
import imutils

PATH_TO_CLIPS = "vods/"

"""### Code Functions"""

def parse_video(file_name, full_path, original_count, label):
    count = 0

    video = cv2.VideoCapture(full_path)
    success, frame = video.read()

    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Loading %s (%s), with FPS %d and total frame count %d ' % (file_name, success, fps, total_frame_count))
    while success:
        # original video at 60FPS, FPS = (1/3) * FPS to reduce redundancy in training data.
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
        if "test" in file_name:
            cv2.imwrite( "data/" + label + "_test"+ '/' + str(original_count * 15 + count) + '.jpg', frame)
        else:
            cv2.imwrite("data/" + label + "_train" + '/' + str(original_count * 15 + count) + '.jpg', frame)

    video.release()

def convert_clips():
    for clip_name in os.listdir(PATH_TO_CLIPS):
        if ".mp4" not in clip_name:
            continue
        # the name of the video holds the label, ex soldier_1.mp4, genji_7.mp4, etc.
        #clip_name = clip_name.split(".mp4")[0]
        label = clip_name.split("_")[0]
        print("Ult Charge: ", label)
        # temporary
        if not os.path.isdir("data/" + label + "_train"):
            os.mkdir("data/" + label + "_train")

        if not os.path.isdir("data/" + label + "_test"):
            os.mkdir("data/" + label + "_test")

        # test files get placed in a different folder.
        if "test" in clip_name:
            parse_video(clip_name, PATH_TO_CLIPS + clip_name, len(os.listdir("data/" + label + "_test")), label)
        else:
            parse_video(clip_name, PATH_TO_CLIPS + clip_name, len(os.listdir("data/" + label + "_train")), label)
    print("Conversion complete!")

if __name__ == '__main__':
    convert_clips()
