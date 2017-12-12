#!/usr/bin/env python

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

import dlib


def get_faces():
    detector = dlib.get_frontal_face_detector()
    # face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')

    for img_file in glob.glob('./face2016/*.jpg'):
        print(img_file)
        img = plt.imread(img_file)
        if (img.shape[0] * img.shape[1]) < (200 * 200):
            continue
        scale = np.sqrt((500. * 500.) / (img.shape[0] * img.shape[1]))
        img = cv2.resize(img, None, None, fx=scale, fy=scale)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        dets = detector(img, 1)
        if len(dets) != 1:
            continue

        H, W = img.shape[:2]
        for d in dets:
            x1 = min(max(d.left(), 0), W)
            x2 = min(max(d.right(), 0), W)
            y1 = min(max(d.top(), 0), H)
            y2 = min(max(d.bottom(), 0), H)

            # enlarge bbox
            cx = (x1 + x2) / 2.
            cy = (y1 + y2) / 2.
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            bbox_w *= 2.0
            bbox_h *= 2.0
            cy -= (bbox_h * 0.1)
            x1 = cx - (bbox_w / 2.)
            x2 = x1 + bbox_w
            y1 = cy - (bbox_h / 2.)
            y2 = y1 + bbox_h
            x1 = min(max(x1, 0), W)
            x2 = min(max(x2, 0), W)
            y1 = min(max(y1, 0), H)
            y2 = min(max(y2, 0), H)
            y1, x1, y2, x2 = map(int, [y1, x1, y2, x2])

            yield (y1, x1, y2, x2), img


if __name__ == '__main__':
    import mvtk
    for bbox, img in get_faces():
        y1, x1, y2, x2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        mvtk.io.plot_tile([img])
        mvtk.io.show()
