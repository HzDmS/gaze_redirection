from imutils import face_utils
import numpy as np
import argparse
import dlib
import cv2
import glob
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, help='path of face images')
parser.add_argument('--trg', type=str, help='path of destination folder')
parser.add_argument('--predictor', type=str, help='path of face landmarks')

args = parser.parse_args()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args.predictor)
# fa = face_utils.FaceAligner(predictor, desiredFaceWidth=2048)

if not os.path.exists(args.trg):
    os.mkdir(args.trg)
else:
    print('folder exist!')
    exit(0)

files = glob.glob(os.path.join(args.src, '*.jpg'))

lm_dict = dict()

for f in files:

    if '_0P_' in f:

        file_name = f.split('/')[-1].split('.')[0]
        print(file_name)

        image = cv2.imread(f)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)

        # faceAligned = fa.align(image, gray, rects[0])
        # grayAligned = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)
        # rectsAligned = detector(grayAligned, 1)

        # shape = predictor(faceAligned, rectsAligned[0])
        faceAligned = image
        shape = predictor(faceAligned, rects[0])
        shape = face_utils.shape_to_np(shape)

        for eye in ['left_eye', 'right_eye']:

            i, j = face_utils.FACIAL_LANDMARKS_IDXS[eye]
            (x, y), radius = cv2.minEnclosingCircle(np.array([shape[i: j]]))
            center = (int(x), int(y))
            radius = int(1.7 * radius)
            top = center[1] - radius
            bottom = center[1] + radius
            left = center[0] - radius
            right = center[0] + radius
            patch = faceAligned[top: bottom + 1, left: right + 1]
            # patch = imutils.resize(roi, width=64, inter=cv2.INTER_CUBIC)

            if eye == 'left_eye':
                save_name = file_name + '_L.jpg'
                landmarks = shape[i:j] - [left, top]
            else:
                save_name = file_name + '_R.jpg'
                patch = cv2.flip(patch, 1)

                landmarks = shape[i:j] - [right, top]
                landmarks[:, 0] = -landmarks[:, 0]
                landmarks_tmp = np.copy(landmarks)

                landmarks[0] = landmarks_tmp[3]
                landmarks[1] = landmarks_tmp[2]
                landmarks[2] = landmarks_tmp[1]
                landmarks[3] = landmarks_tmp[0]
                landmarks[4] = landmarks_tmp[5]
                landmarks[5] = landmarks_tmp[4]

            info = np.zeros([landmarks.shape[0], landmarks.shape[1] + 1],
                            dtype=np.int32)
            info[:, :2] = landmarks
            info[:, 2] = 2 * radius + 1
            lm_dict[save_name] = info.tolist()
            cv2.imwrite(os.path.join(args.trg, save_name), patch)

with open(os.path.join(args.trg, 'lm.json'), 'w') as f:

    json.dump(lm_dict, f)
