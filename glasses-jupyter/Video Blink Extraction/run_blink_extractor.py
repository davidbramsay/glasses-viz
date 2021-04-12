from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imutils
import time
import dlib
import cv2
import argparse

# python run_blink_extractor.py --video blink_detection_demo.mp4 --resize-width 720
# python run_blink_extractor.py --video blink_detection_demo.mp4

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

def analyze_video_for_blinks(video_filename, resize_width=720):

    frame_data = {}
    counter = 0

    csv_name = video_filename[:-4] + '_eyeratio.csv'

    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # start the video stream thread
    print("[INFO] starting video stream thread...")
    vs = cv2.VideoCapture(video_filename)

    total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vs.get(cv2.CAP_PROP_FPS)
    print('FPS:',fps,'\tTotal Frames:',total_frames,'\tDuration (s):',total_frames/fps)

    eye_ratio_data = pd.DataFrame()

    time.sleep(1.0)

    # loop over frames from the video stream
    for fno in range(0, total_frames):

        if not fno % (fps*30): print('%8d frames processed (%3.2f min)' % (fno, fno/fps/60))

        _, frame = vs.read()
        frame = imutils.resize(frame, width=resize_width)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_data['timestamp'] = vs.get(cv2.CAP_PROP_POS_MSEC)
        frame_data['eye_ratio'] = np.nan

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # loop over any face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            ear_ratio = ear/w

            if np.isnan(frame_data['eye_ratio']): frame_data['eye_ratio'] = ear_ratio
            elif ear_ratio < frame_data['eye_ratio']: frame_data['eye_ratio'] = ear_ratio

        eye_ratio_data = eye_ratio_data.append(frame_data, ignore_index=True)

    print('[INFO] completed blink analysis.')

    print('[INFO] saving to', csv_name)
    eye_ratio_data.to_csv(csv_name)

    return eye_ratio_data, fps

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, required=True,
	help="path to input video file")
ap.add_argument("-r", "--resize-width", type=int, default=720,
	help="resize width for image before processing")


args = vars(ap.parse_args())
analyze_video_for_blinks(args['video'], args['resize_width'])
