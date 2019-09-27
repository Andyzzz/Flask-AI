#!/usr/bin/python

# Copyright (c) 2015 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
This is the code behind the Switching Eds blog post:

    http://matthewearl.github.io/2015/07/28/switching-eds-with-python/

See the above for an explanation of the code below.

To run the script you'll need to install dlib (http://dlib.net) including its
Python bindings, and OpenCV. You'll also need to obtain the trained model from
sourceforge:

    http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2

Unzip with `bunzip2` and change `PREDICTOR_PATH` to refer to this file. The
script is run like so:

    ./faceswap.py <head image> <face image>

If successful, a file `output.jpg` will be produced with the facial features
from `<head image>` replaced with the facial features from `<face image>`.

"""

import cv2
import dlib
import numpy
import os

import sys


class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass
class Faceswap(object):
    def __init__(self, ):

        # PREDICTOR_PATH = "/home/matt/dlib-18.16/shape_predictor_68_face_landmarks.dat"
        PREDICTOR_PATH = "/data/code/STGAN_latest/STGAN/face_swap/shape_predictor_68_face_landmarks.dat"


        self.SCALE_FACTOR = 1
        self.FEATHER_AMOUNT = 11

        self.FACE_POINTS = list(range(17, 68))
        self.MOUTH_POINTS = list(range(48, 61))
        self.RIGHT_BROW_POINTS = list(range(17, 22))
        self.LEFT_BROW_POINTS = list(range(22, 27))
        self.RIGHT_EYE_POINTS = list(range(36, 42))
        self.LEFT_EYE_POINTS = list(range(42, 48))
        self.NOSE_POINTS = list(range(27, 35))
        self.AW_POINTS = list(range(0, 17))

        self.Marvel_Faces = {
                "1marvel005.jpg":[646,45,930,347,],
                "1marvel006.jpg": [511, 223, 953, 637],
                "1marvel007.jpg":[793,19,1102,415,],
                "1marvel008.jpg":[1176,59,1523,510,],
                "1marvel012.jpg":[922,14,1243,401,],
                "1marvel013.jpg":[896,6,1268,410,],
                "1marvel018.jpg":[559,94,1000,513,],
                "1marvel019.jpg":[558,49,1028,505,],
                "1marvel020.jpg":[1020,20,1335,420,],
                "1marvel022.jpg":[587,18,941,453,],
                "1marvel023.jpg":[1028,23,1391,468,],
                "2marvel11.jpg":[518,1,1180,683,],
                "2marvel12.jpg":[418,36,813,465,],
                "2marvel14.jpg":[1077,44,1357,387,],
                "2marvel15.jpg":[1000,10,1675,715,],
                "2marvel16.jpg":[557,28,959,541,],
                "2marvel17.jpg":[504,205,740,479,],
                "2marvel19.jpg":[858,209,1072,523,],
                "2marvel2.jpg":[671,79,1003,497,],
                "2marvel22.jpg":[578,3,1153,523,],
                "2marvel23.jpg":[631,61,1095,565,],
                "2marvel26.jpg":[1077,1,1498,563,],
                "2marvel29.jpg":[1154,59,1328,274,],
                "2marvel3.jpg":[618,33,970,476,],
                "2marvel30.jpg":[539,24,907,450,],
                "2marvel34.jpg":[1090,17,1520,585,],
                "2marvel41.png":[404,15,573,195,],
                "2marvel5.jpg":[962,5,1605,687,],
                "2marvel8.jpg":[959,167,1145,368,],
                "2marvel9.jpg":[1136,41,1457,461,],
                "5aoyun.jpg": [360, 260, 545, 518],
                "4gaigekaifang.jpg": [603, 121, 703, 263],
                "2kangmeiyuanchao.jpg": [247, 442, 340, 570],
                "1kaiguodadian.jpg": [746, 264, 999, 560],
                "3liangdanyixing.jpg": [1297, 169, 1685, 680],
                "6yidaiyilu.jpg": [611, 162, 678, 239],  # [229, 198, 314, 294]
                '2013.jpg': [676, 206, 748, 299],
                '2015.jpg': [246, 215, 317, 306],
                '2016.jpg': [633, 119, 682, 181],
                '2017.jpg': [217, 70, 265, 122],
                '2019.jpg': [408, 141, 473, 228]
                }

        self.Subtitle = {
            "marvel17.jpg": '',
            "marvel19.jpg": '',
            "marvel21.jpg": '',
            "marvel41.png": '',
            "marvel006.jpg": '',
            "marvel012.jpg": '',
            "marvel013.jpg": '',
            "5aoyun.jpg": '2008年北京奥运会开幕式上，你作为一名打击乐手参与其中，开幕式前的刻苦训练成就了你此时的飒爽英姿。这是中国首次承办夏季奥运会，你展露出身为一个大国国民的自豪与幸福感。',
            "4gaigekaifang.jpg": '1992年，你随从邓小平总书记进行南方视察，见证进一步改革开放的浪潮开始。',
            "2kangmeiyuanchao.jpg": '你在朝鲜经过艰苦斗争，赶走了美国侵略者，迎来了抗美援朝战争的胜利，凯旋归来！',
            "1kaiguodadian.jpg": '1949年，你在天安门广场参加开国大典，见证新中国成立，外国侵略者被赶出中国，中国人民迎来了新的时代，你的内心激动无比。',
            "3liangdanyixing.jpg": '你参与了氢弹的研发。当你接到研发氢弹的任务时，你明白这其中的困难，更加懂得这对于国家的重要性。在这两年零八个月的日夜里你不眠不休，攻坚克难。终于，蘑菇云在中国西部升起时，全世界都感到震惊。那一刻，你为之动容。',
            "6yidaiyilu.jpg": '你随从习近平主席出访哈萨克斯坦，商讨“一带一路”事宜，感受到大国的开放和责任感。',
            '2013.jpg': '2013年，偶然的机会，你得到了一张春节联欢晚会的门票，这是你第一次在现场观看春晚，更加幸运的是，你还碰巧出现在了刘谦表演魔术的镜头里。',
            '2015.jpg': '2015年，13年观看春晚的经历让你十分难忘，但只能坐在台下观看也让你感到遗憾，于是，你高喊着“我要上春晚”报名了春晚的群众演员。凭借着扎实的表演功底，你成功入选了孙涛，邵峰的小品《社区民警于三快》，第一次登上了春晚的舞台。',
            '2016.jpg': '2016年，你所在的公司遭遇金融危机，亏欠你几个月的工资，没钱回家过年的你只能再次报名春晚群演，好挣些钱回家过年，这一次，一个开场镜头纪录下了你。',
            '2017.jpg': '2017年，参加了三次春晚让你仿佛对春晚产生了感情，在这次的春晚舞台上，你为tf-boys伴舞，收获了一次从未有过的经历。',
            '2019.jpg': '2019年，听说李易峰和朱一龙要在春晚的舞台上打篮球，作为花式篮球高级玩家的你再也坐不住了，今年春晚的舞台上，你与他们完成了一场令人难忘的篮球大秀。'
        }

        self.Marvel_Gesture = {
            "marvel1.jpg": 'left',
            "marvel2.jpg": 'mid',
            "marvel3.jpg": 'right',
            "marvel5.jpg": 'right',
            "marvel8.jpg": 'left_watch',
            "marvel9.jpg": 'shock',
            "marvel11.jpg": 'shock',
            "marvel12.jpg": 'mid',
            "marvel14.jpg": 'shock',
            "marvel15.jpg": 'shock',
            "marvel16.jpg": 'mid',
            "marvel17.jpg": 'left',
            "marvel18.jpg": 'mid',
            "marvel19.jpg": 'mid',
            "marvel21.jpg": 'mid',
            "marvel22.jpg": 'frown',
            "marvel23.jpg": 'mid',
            "marvel26.jpg": 'right',
            "marvel28.jpg": 'left',
            "marvel29.jpg": 'mid',
            "marvel30.jpg": 'mid',
            "marvel31.jpg": 'mid',
            "marvel34.jpg": 'mid',
            "marvel41.png": 'mid',
            "marvel001.jpg":'mid',
            "marvel003.jpg": 'mid',
            "marvel004.jpg": 'mid',
            "marvel005.jpg": 'shock',
            "marvel006.jpg": 'mid1',
            "marvel007.jpg": 'left',
            "marvel008.jpg": 'right',
            "marvel009.jpg": 'right',
            "marvel010.jpg": 'mid',
            "marvel012.jpg": 'mid',
            "marvel013.jpg": 'mid',
            "marvel015.jpg": 'right',
            "marvel016.jpg": 'right',
            "marvel017.jpg": 'left',
            "marvel018.jpg": 'mid',
            "marvel020.jpg": 'shock',
            "marvel023.jpg": 'right_open',
            "marvel022.jpg": 'mid',

        }

        # Points used to line up the images.
        self.ALIGN_POINTS = (self.LEFT_BROW_POINTS + self.RIGHT_EYE_POINTS + self.LEFT_EYE_POINTS +
                                       self.RIGHT_BROW_POINTS + self.NOSE_POINTS + self.MOUTH_POINTS)

        # Points from the second image to overlay on the first. The convex hull of each
        # element will be overlaid.
        self.OVERLAY_POINTS = [
            self.LEFT_EYE_POINTS + self.RIGHT_EYE_POINTS + self.LEFT_BROW_POINTS + self.RIGHT_BROW_POINTS,
            self.NOSE_POINTS + self.MOUTH_POINTS,
        ]

        # Amount of blur to use during colour correction, as a fraction of the
        # pupillary distance.
        self.COLOUR_CORRECT_BLUR_FRAC = 0.6

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(PREDICTOR_PATH)


    def get_landmarks(self,im):
        rects = self.detector(im, 1)

        if len(rects) > 1:
            raise TooManyFaces
        if len(rects) == 0:
            raise NoFaces

        return numpy.matrix([[p.x, p.y] for p in self.predictor(im, rects[0]).parts()])

    def annotate_landmarks(self,im, landmarks):
        im = im.copy()
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            cv2.putText(im, str(idx), pos,
                        fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 255))
            cv2.circle(im, pos, 3, color=(0, 255, 255))
        return im

    def draw_convex_hull(self,im, points, color):
        points = cv2.convexHull(points)
        cv2.fillConvexPoly(im, points, color=color)

    def get_face_mask(self,im, landmarks):
        im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

        for group in self.OVERLAY_POINTS:
            self.draw_convex_hull(im,
                             landmarks[group],
                             color=1)

        im = numpy.array([im, im, im]).transpose((1, 2, 0))

        im = (cv2.GaussianBlur(im, (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT), 0) > 0) * 1.0
        im = cv2.GaussianBlur(im, (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT), 0)

        return im

    def transformation_from_points(self,points1, points2):
        """
        Return an affine transformation [s * R | T] such that:

            sum ||s*R*p1,i + T - p2,i||^2

        is minimized.

        """
        # Solve the procrustes problem by subtracting centroids, scaling by the
        # standard deviation, and then using the SVD to calculate the rotation. See
        # the following for more details:
        #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

        points1 = points1.astype(numpy.float64)
        points2 = points2.astype(numpy.float64)

        c1 = numpy.mean(points1, axis=0)
        c2 = numpy.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2

        s1 = numpy.std(points1)
        s2 = numpy.std(points2)
        points1 /= s1
        points2 /= s2

        U, S, Vt = numpy.linalg.svd(points1.T * points2)

        # The R we seek is in fact the transpose of the one given by U * Vt. This
        # is because the above formulation assumes the matrix goes on the right
        # (with row vectors) where as our solution requires the matrix to be on the
        # left (with column vectors).
        R = (U * Vt).T

        return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                           c2.T - (s2 / s1) * R * c1.T)),
                             numpy.matrix([0., 0., 1.])])

    def read_im_and_landmarks(self,fname):
        im = cv2.imread(fname,cv2.IMREAD_COLOR )#
        print(numpy.shape(im))
        im = cv2.resize(im, (im.shape[1] * self.SCALE_FACTOR,
                             im.shape[0] * self.SCALE_FACTOR))
        s = self.get_landmarks(im)

        return im, s

    def warp_im(self,im, M, dshape):
        output_im = numpy.zeros(dshape, dtype=im.dtype)
        cv2.warpAffine(im,
                       M[:2],
                       (dshape[1], dshape[0]),
                       dst=output_im,
                       borderMode=cv2.BORDER_TRANSPARENT,
                       flags=cv2.WARP_INVERSE_MAP)
        return output_im

    def correct_colours(self,im1, im2, landmarks1):
        blur_amount = self.COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
                                  numpy.mean(landmarks1[self.LEFT_EYE_POINTS], axis=0) -
                                  numpy.mean(landmarks1[self.RIGHT_EYE_POINTS], axis=0))
        blur_amount = int(blur_amount)
        if blur_amount % 2 == 0:
            blur_amount += 1
        im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
        im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

        # Avoid divide-by-zero errors.
        im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

        return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
                                                    im2_blur.astype(numpy.float64))

    def swap(self,ori,path):
        filename1 = path
        filename2 = ori

        # im1, landmarks1 = read_im_and_landmarks(sys.argv[1])
        # im2, landmarks2 = read_im_and_landmarks(sys.argv[2])
        print(filename1, filename2)
        im1, landmarks1 = self.read_im_and_landmarks(filename1)

        im2, landmarks2 = self.read_im_and_landmarks(filename2)

        M = self.transformation_from_points(landmarks1[self.ALIGN_POINTS],
                                       landmarks2[self.ALIGN_POINTS])

        mask = self.get_face_mask(im2, landmarks2)
        warped_mask = self.warp_im(mask, M, im1.shape)
        combined_mask = numpy.max([self.get_face_mask(im1, landmarks1), warped_mask],
                                  axis=0)

        warped_im2 = self.warp_im(im2, M, im1.shape)
        warped_corrected_im2 = self.correct_colours(im1, warped_im2, landmarks1)

        output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
        return(output_im)

    # cv2.imwrite('output.jpg', output_im)

    def swap_part(self,ori,path,labels):
        filename1 = path
        filename2 = ori
        # xmin=2257
        # ymin=1331
        # xmax=2532
        # ymax=1606
        xmin,ymin,xmax,ymax = labels
        # im1, landmarks1 = read_im_and_landmarks(filename1)
        im = cv2.imread(filename1, cv2.IMREAD_COLOR)
        im_part = im[ymin:ymax,xmin:xmax,:]

        print(im.shape)
        im1 = cv2.resize(im_part, (im_part.shape[1] * self.SCALE_FACTOR,
                             im_part.shape[0] * self.SCALE_FACTOR))
        landmarks1 = self.get_landmarks(im1)

        # im1, landmarks1 = read_im_and_landmarks(filename1)
        im2, landmarks2 = self.read_im_and_landmarks(filename2)

        M = self.transformation_from_points(landmarks1[self.ALIGN_POINTS],
                                       landmarks2[self.ALIGN_POINTS])

        mask = self.get_face_mask(im2, landmarks2)
        warped_mask = self.warp_im(mask, M, im1.shape)
        combined_mask = numpy.max([self.get_face_mask(im1, landmarks1), warped_mask],
                                  axis=0)

        warped_im2 = self.warp_im(im2, M, im1.shape)
        warped_corrected_im2 = self.correct_colours(im1, warped_im2, landmarks1)

        output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
        for i in range(ymin,ymax):
            for j in range(xmin,xmax):
                im[i][j] = output_im[i-ymin][j-xmin]
        return(im)
    def test(self,):
        # 遍历
        rootdir = './marvel'
        ori = './41n'
        list_ori = os.listdir(ori)
        list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
        for i in range(0, len(list_ori)):
            path_ori = ori + "/" + list_ori[i]
            path1 = './result' + "/" + list_ori[i][:-4]
            print(path1)
            isExists = os.path.exists(path1)
            # 判断结果
            if not isExists:
                os.mkdir(path1)

            for i in range(0, len(list)):
                # print(,list[i])
                # path = os.path.join(rootdir, list[i])
                path = rootdir + "/" + list[i]
                if os.path.isfile(path):
                    output = self.swap(path_ori, path)
                    print(numpy.shape(output))
                    cv2.imwrite(path1 + "/" + str(i) + '.jpg', output)

        # Read images

        # cv2.imwrite('result.jpg', output)
        #
        # cv2.imshow("Face Swapped", output)
        # cv2.waitKey(0)
        print('finish')
        return

    def test_a_person(self,):
        # 遍历
        rootdir = 'C:/Users/rht/PycharmProjects/face_swap/mar'
        path_ori = 'C:/Users/rht/PycharmProjects/face_swap/41n/lgz'

        list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件

        for i in range(0, len(list)):
            print(list[i])
            # path = os.path.join(rootdir, list[i])
            path = rootdir + "/" + list[i]
            if os.path.isfile(path):
                print(path)
                labels = self.Marvel_Faces.get(list[i], 'single_face')
                gesture = self.Marvel_Gesture.get((list[i]), 'fail')
                if gesture != 'fail':
                    path_ori_gesture = path_ori + '_' + gesture + ".jpg"
                else:
                    print(gesture)
                if labels == "single_face":

                    output = self.swap(path_ori_gesture, path)
                else:
                    output = self.swap_part(path_ori_gesture,path,labels)
                # print(numpy.shape(output))
                print('./result/person_result' + "/" + 'lgz'+str(i) + '.jpg has been saved')
                cv2.imwrite('./result/person_result2' + "/" + list[i], output)
        print('finish')
        return
    def swap1(self,img,path,im2,landmarks2):
        filename1 = path
        # filename2 = ori

        # im1, landmarks1 = read_im_and_landmarks(sys.argv[1])
        # im2, landmarks2 = read_im_and_landmarks(sys.argv[2])
        # print(filename1, filename2)
        im1, landmarks1 = self.read_im_and_landmarks(filename1)
        # im = cv2.imread(img, cv2.IMREAD_COLOR)  #
        # print(numpy.shape(im))

        # im2, landmarks2 = self.read_im_and_landmarks(filename2)

        M = self.transformation_from_points(landmarks1[self.ALIGN_POINTS],
                                       landmarks2[self.ALIGN_POINTS])

        mask = self.get_face_mask(im2, landmarks2)
        warped_mask = self.warp_im(mask, M, im1.shape)
        combined_mask = numpy.max([self.get_face_mask(im1, landmarks1), warped_mask],
                                  axis=0)

        warped_im2 = self.warp_im(im2, M, im1.shape)
        warped_corrected_im2 = self.correct_colours(im1, warped_im2, landmarks1)

        output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
        return(output_im)

    # cv2.imwrite('output.jpg', output_im)

    def swap_part1(self,img,path,labels,im2,landmarks2):
        filename1 = path
        # filename2 = ori
        # xmin=2257
        # ymin=1331
        # xmax=2532
        # ymax=1606
        xmin,ymin,xmax,ymax = labels
        # im1, landmarks1 = read_im_and_landmarks(filename1)
        im = cv2.imread(filename1, cv2.IMREAD_COLOR)
        im_part = im[ymin:ymax,xmin:xmax,:]

        print(im.shape)
        im1 = cv2.resize(im_part, (im_part.shape[1] * self.SCALE_FACTOR,
                             im_part.shape[0] * self.SCALE_FACTOR))
        landmarks1 = self.get_landmarks(im1)

        # im1, landmarks1 = read_im_and_landmarks(filename1)
        # im2, landmarks2 = self.read_im_and_landmarks(filename2)
        # im = cv2.imread(img, cv2.IMREAD_COLOR)  #
        # print(numpy.shape(im))
        # im2 = cv2.resize(img, (img.shape[1] * self.SCALE_FACTOR,
        #                       img.shape[0] * self.SCALE_FACTOR))
        # landmarks2 = self.get_landmarks(im2)

        M = self.transformation_from_points(landmarks1[self.ALIGN_POINTS],
                                       landmarks2[self.ALIGN_POINTS])

        mask = self.get_face_mask(im2, landmarks2)
        warped_mask = self.warp_im(mask, M, im1.shape)
        combined_mask = numpy.max([self.get_face_mask(im1, landmarks1), warped_mask],
                                  axis=0)

        warped_im2 = self.warp_im(im2, M, im1.shape)
        warped_corrected_im2 = self.correct_colours(im1, warped_im2, landmarks1)

        output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
        for i in range(ymin,ymax):
            for j in range(xmin,xmax):
                im[i][j] = output_im[i-ymin][j-xmin]
        return(im)
    def test_online(self,img,path):
        rootdir = path
        # path_ori = 'C:/Users/rht/PycharmProjects/face_swap/41n/lgz'

        list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
        outputs = []
        subtitles = []
        im2 = cv2.resize(img, (img.shape[1] * self.SCALE_FACTOR,
                               img.shape[0] * self.SCALE_FACTOR))
        landmarks2 = self.get_landmarks(im2)
        for i in range(0, len(list)):
            print(list[i])
            # path = os.path.join(rootdir, list[i])
            path = rootdir + "/" + list[i]
            if os.path.isfile(path):
                print(path)
                labels = self.Marvel_Faces.get(list[i], 'single_face')
                subtitle = self.Subtitle.get(list[i], '')
                # gesture = self.Marvel_Gesture.get((list[i]), 'fail')
                if labels == "single_face":
                    output = self.swap1(img, path,im2,landmarks2)
                else:
                    output = self.swap_part1(img, path, labels,im2,landmarks2)
                # print(numpy.shape(output))
                print('./result/person_result' + "/" + 'lgz' + str(i) + '.jpg has been saved')

                outputs.append(output)
                subtitles.append(subtitle)
                # cv2.imwrite('./result/online' + "/" + list[i], output)
        print('finish')
        return outputs,subtitles
if __name__ == '__main__':
    # path_ori = './41n/lyp.jpg'
    # path = './mar/marvel9.jpg'
    # output = swap(path_ori, path)
    # cv2.imshow('a',output)
    # cv2.waitKey()
    # # print(numpy.shape(output))
    # path1 = './result/dee/1.jpg'
    # cv2.imwrite(path1, output)
    a = Faceswap()
    path1 = './41n/lgz2.jpg'
    path = './SpringFestival'
    img = cv2.imread(path1, cv2.IMREAD_COLOR)
    outputs,subtitles = a.test_online(img,path)
