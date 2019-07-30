from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, help='path of face images')
parser.add_argument('--trg', type=str, help='path of destination folder')
parser.add_argument('--predictor', type=str, help='path of face landmarks')

args = parser.parse_args()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args.predictor)
fa = face_utils.FaceAligner(predictor, desiredFaceWidth=512)

if not os.path.exists(args.trg):
    os.mkdir(args.trg)
else:
    print('folder exist!')
    quit()

files = glob.glob(os.path.join(args.src, '*/*.jpg'))

for f in files:

    if '_0P_' in f:

        file_name = f.split('/')[-1]
        print(file_name)

        image = cv2.imread(f)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)

        faceAligned = fa.align(image, gray, rects[0])
        cv2.imwrite(os.path.join(args.trg, file_name), faceAligned)
