# -*- coding: utf-8 -*-
'''
@Author:    Kabuto
@Date:      2019.07.25
@Purpose:   Detect face from a specific image.
            Square area
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt


def faceDetect(imgPath, saveName, imgSize=(256, 256)):
    '''
    Detect face from a specific image
    :param imgPath:     The path of the image
    :param imgSize:     The return size of the face picture
    :return:            return the specific size of the face image
    '''
    # read an image
    img = cv2.imread(imgPath)
    # set face detector
    face_cascade = cv2.CascadeClassifier('opencv_classifier_files\\haarcascade_frontalface_alt2.xml')
    face = face_cascade.detectMultiScale(img, 1.1, 5)
    # set return path
    savePath = "save/"
    if len(face):
        for (x, y, w, h) in face:
            # find the center of the face
            x_center = int(x + 0.5 * w)
            y_center = int(y + 0.5 * h)
            # the max width of the face area
            face_width = int(w / 2 * 1.7)
            # find x_min, x_max, y_min, y_max
            x_min = max(0, x_center - face_width)
            x_max = min(x_center + face_width, img.shape[1])
            y_min = max(0, y_center - face_width)
            y_max = min(y_center + face_width, img.shape[0])
            if x_min == 0:
                x_max = 2 * face_width
            if x_max == img.shape[1]:
                x_min = img.shape[1] - 2 * face_width
            if y_min == 0:
                y_max = 2 * face_width
            if y_max == img.shape[0]:
                y_min = img.shape[0] - 2 * face_width

            # print(x, y, w, h, "area:", x_min, x_max, y_min, y_max)
            # col = int(x * 0.5)
            # # W = min(int((x + w) * 1.5), img.shape[1])
            # W = min(int(x*2 + w), img.shape[1])
            # row = int(y * 0.5)
            # # H = min(int((y + h) * 1.5), img.shape[0])
            # H = min(int(y*2 + h), img.shape[0])
            # # f = cv2.resize(img[Y:H, X:W], imgSize)

            f = cv2.resize(img[y_min:y_max, x_min:x_max], imgSize)
            cv2.imwrite(savePath + '{0}.jpg'.format(saveName), f)
    else:
        print("No face is detected!")


if __name__ == '__main__':
    # for i in range(1, 16):
    #     faceDetect("image\\{0}.jpg".format(i), "face"+str(i))
    # print("Over!")
    faceDetect("lgz.jpg", "lgz", imgSize=(128, 128))
