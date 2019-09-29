# -*- coding: utf-8 -*-
'''
@Author:    Kabuto
@Date:      2019.07.28
@Purpose:   Detect face from a specific image.
            Rectangle area
'''

import cv2


def faceDetect(imgPath, savePath, imgSize=(256, 256)):
    '''
    Detect face from a specific image
    :param imgPath:     The path of the image
    :param imgSize:     The return size of the face picture
    :return:            return the specific size of the face image
    '''
    # read an image
    img = cv2.imread(imgPath)
    if min(img.shape[0], img.shape[1]) > 1024:
        p = max(img.shape[0], img.shape[1]) / 1024
        img = cv2.resize(img, (int(img.shape[1] / p), int(img.shape[0] / p)))
    # set face detector
    face_cascade = cv2.CascadeClassifier(
        'G:/Deecamp/STGAN_ZW/STGAN/FaceDetector/opencv_classifier_files/haarcascade_frontalface_alt2.xml')
    # face = face_cascade.detectMultiScale(img, 1.1, 10)scaleFactor=None, minNeighbors=None,
    face = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=10)

    if len(face):
        for (x, y, w, h) in face:
            # find the center of the face
            x_center = int(x + 0.5 * w)
            y_center = int(y + 0.35 * h)
            # the max width of the face area
            zoom_rate = 1.7
            face_width = int(w / 2 * zoom_rate)
            face_height = int(w * imgSize[1] / imgSize[0] / 2 * zoom_rate)
            # find x_min, x_max, y_min, y_max
            x_min = max(0, x_center - face_width)
            x_max = min(x_center + face_width, img.shape[1])
            y_min = max(0, y_center - face_height)
            y_max = min(y_center + face_height, img.shape[0])
            if x_min == 0:
                x_max = 2 * face_width
            if x_max == img.shape[1]:
                x_min = img.shape[1] - 2 * face_width
            if y_min == 0:
                y_max = 2 * face_height
            if y_max == img.shape[0]:
                y_min = img.shape[0] - 2 * face_height

            f = cv2.resize(img[y_min:y_max, x_min:x_max], imgSize)
            # print(np.shape(f))
            # print(savePath)
            cv2.imwrite(savePath, f)
    else:
        print("No face is detected!")


def faceDetect_server(img, imgSize):
    '''
    Detect face from a specific image
    :param imgPath:     The path of the image
    :param imgSize:     The return size of the face picture
    :return:            return the specific size of the face image
    '''
    # read an image
    if min(img.shape[0], img.shape[1]) > 1024:
        p = max(img.shape[0], img.shape[1]) / 1024
        img = cv2.resize(img, (int(img.shape[1] / p), int(img.shape[0] / p)))
    # set face detector

    # face_cascade = cv2.CascadeClassifier(
    #     'G:/Deecamp/STGAN_ZW/STGAN/FaceDetector/opencv_classifier_files/haarcascade_frontalface_alt2.xml')
    face_cascade = cv2.CascadeClassifier(
        'G:/PythonProjects/LearnSamples/Flask-AI/FaceDetector/opencv_classifier_files/haarcascade_frontalface_alt2.xml')
    # 这里要用绝对路径，否则会找不到文件
    # face = face_cascade.detectMultiScale(img, 1.1, 10)
    face = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=10)
    print(len(face))
    if len(face):
        for (x, y, w, h) in face:
            # find the center of the face
            x_center = int(x + 0.5 * w)
            y_center = int(y + 0.35 * h)
            # the max width of the face area
            zoom_rate = 1.7  # 1.7
            face_width = int(w / 2 * zoom_rate)
            face_height = int(w * imgSize[1] / imgSize[0] / 2 * zoom_rate)
            # find x_min, x_max, y_min, y_max
            x_min = max(0, x_center - face_width)
            x_max = min(x_center + face_width, img.shape[1])
            y_min = max(0, y_center - face_height)
            y_max = min(y_center + face_height, img.shape[0])
            if x_min == 0:
                x_max = 2 * face_width
            if x_max == img.shape[1]:
                x_min = img.shape[1] - 2 * face_width
            if y_min == 0:
                y_max = 2 * face_height
            if y_max == img.shape[0]:
                y_min = img.shape[0] - 2 * face_height

            finalImg = cv2.resize(img[y_min:y_max, x_min:x_max], imgSize)
            # print('resized shape: ', f.shape)
            return finalImg
    else:
        return None


if __name__ == '__main__':
    img = cv2.imread('G:\\Deecamp\\cxk.png')
    # img = cv2.imread('G:\\Deecamp\\STGAN_ZW\\STGAN\\transform_folder\\cxk.png')
    result = faceDetect_server(img, (128, 128))
    cv2.imwrite('G:/Deecamp/test0_face_detect.png', result)


