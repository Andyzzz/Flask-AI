# -*- coding=utf-8 -*-
import math
import warnings

warnings.filterwarnings("ignore")
import cv2
import numpy as np
# import sys
# sys.path.append('./face_toolbox_keras/')
from segmodels.parser import face_parser
from PIL import Image
from imageio import imread, imsave


class changeFace():
    def __init__(self):
        # self.im = im
        # self.change_name = change_name
        # self.tgtRGB = tgtRGB
        # self.mix = mix
        self.prs = face_parser.FaceParser()

    def segmentation(self, im):
        # prs = face_parser.FaceParser()
        out = self.prs.parse_face(im)
        seg = out[0]
        # cv2.imwrite('G:/Deecamp/segment.png', seg)
        return seg

    def resize_image(self, im, max_size=768):
        if np.max(im.shape) > max_size:
            ratio = max_size / np.max(im.shape)
            # print(f"Resize image to ({str(int(im.shape[1] * ratio))}, {str(int(im.shape[0] * ratio))}).")
            return cv2.resize(im, (0, 0), fx=ratio, fy=ratio)
        return im

    def hsv2rgb(self, h, s, v):
        h = float(h)
        s = float(s)
        v = float(v)
        h60 = h / 60.0
        h60f = math.floor(h60)
        hi = int(h60f) % 6
        f = h60 - h60f
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        r, g, b = 0, 0, 0
        if hi == 0:
            r, g, b = v, t, p
        elif hi == 1:
            r, g, b = q, v, p
        elif hi == 2:
            r, g, b = p, v, t
        elif hi == 3:
            r, g, b = p, q, v
        elif hi == 4:
            r, g, b = t, p, v
        elif hi == 5:
            r, g, b = v, p, q
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        return [r, g, b]

    def rgb2hsv(self, r, g, b):
        r, g, b = r / 255.0, g / 255.0, b / 255.0
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx - mn
        if mx == mn:
            h = 0
        elif mx == r:
            h = (60 * ((g - b) / df) + 360) % 360
        elif mx == g:
            h = (60 * ((b - r) / df) + 120) % 360
        elif mx == b:
            h = (60 * ((r - g) / df) + 240) % 360
        if mx == 0:
            s = 0
        else:
            s = df / mx
        v = mx
        return [h, s, v]

    def getAve(self, mask, im):
        Rarea = mask * im[:, :, 0]
        Garea = mask * im[:, :, 1]
        Barea = mask * im[:, :, 2]

        num = np.sum(mask > 0)
        r_sum = Rarea.sum() / 255.0
        g_sum = Garea.sum() / 255.0
        b_sum = Barea.sum() / 255.0

        r_ave = int(r_sum / num)
        g_ave = int(g_sum / num)
        b_ave = int(b_sum / num)

        h, s, v = self.rgb2hsv(r_ave, g_ave, b_ave)
        # print('rgb: ',r_ave,g_ave,b_ave)
        # print('ave hsv: ',h, s, v)
        return [h, s, v]

    def getMask(self, im, changePart):
        seg = self.segmentation(im)
        # segDict =  {'background':[0], 'skin':[1], 'left eyebrow':[2], 'right eyebrow':[3], 'left eye':[4], 'right eye':[5], 'glasses':[6],
        # 'left ear':[7],'right ear':[8], 'earings':[9], 'nose':[10], 'mouth':[12,13], 'upper lip':[12], 'lower lip':[13], 'neck':[14],
        # 'neck_l':[15],'cloth':[16], 'hair':[17], 'hat':[18], 'eyes':[4,5],'ears':[7,8],'eyebrows':[2,3]
        # }
        segDict = {'background': [0], 'skin': [1, 7, 8, 9, 10, 14, 15, ], 'ears': [7, 8, 9], 'eyebrows': [2, 3],
                   'glasses': [6], 'nose': [10], 'mouth': [12, 13], 'neck': [14], 'hair': [17], 'hat': [18],
                   'eyes': [4, 5],
                   }
        h, w, c = im.shape

        mask = np.zeros((h, w))
        for i in segDict[changePart]:
            mask[seg == i] = 255
        print(mask.shape)
        return mask

    def change2(self, im, changePart, tgtRGB, mix=0.6):
        seg = self.segmentation(im)
        # segDict =  {'background':[0], 'skin':[1], 'left eyebrow':[2], 'right eyebrow':[3], 'left eye':[4], 'right eye':[5], 'glasses':[6],
        # 'left ear':[7],'right ear':[8], 'earings':[9], 'nose':[10], 'mouth':[12,13], 'upper lip':[12], 'lower lip':[13], 'neck':[14],
        # 'neck_l':[15],'cloth':[16], 'hair':[17], 'hat':[18], 'eyes':[4,5],'ears':[7,8],'eyebrows':[2,3]
        # }
        segDict = {'background': [0], 'skin': [1, 7, 8, 9, 10, 14, 15, ], 'ears': [7, 8, 9], 'eyebrows': [2, 3],
                   'glasses': [6], 'nose': [10], 'mouth': [12, 13], 'neck': [14], 'hair': [17], 'hat': [18],
                   'eyes': [4, 5],
                   }
        h, w, c = im.shape
        res = im.copy()
        mask = np.zeros((h, w))
        for i in segDict[changePart]:
            mask[seg == i] = 255
        mask2 = mask.copy()
        mask = cv2.blur(mask, (5, 5))

        h_, s_, v_ = self.getAve(mask, im)
        for row in range(0, h):
            for col in range(0, w):
                if mask[row, col] > 0 and seg[row, col] in segDict[changePart]:
                    newmix = mix * mask[row, col] / 255.0
                    srcRGB = [res[row, col, 0], res[row, col, 1], res[row, col, 2]]
                    srcHSV = self.rgb2hsv(srcRGB[0], srcRGB[1], srcRGB[2])
                    tgtHSV = self.rgb2hsv(tgtRGB[0], tgtRGB[1], tgtRGB[2])
                    if v_ < 0.1:
                        srcHSV[2] += 0.3
                        if srcHSV[2] > 1.0:
                            srcHSV[2] = 0.99
                    srcHSVnew = [tgtHSV[0], tgtHSV[1], srcHSV[2]]
                    srcRGBnew = self.hsv2rgb(srcHSVnew[0], srcHSVnew[1], srcHSVnew[2])
                    resRGB = (1 - newmix) * np.array(srcRGB) + newmix * np.array(srcRGBnew)
                    res[row, col, 0], res[row, col, 1], res[row, col, 2] = int(resRGB[0]), int(resRGB[1]), int(
                        resRGB[2])
        return res, mask2

    def change_hair(self, im, seg, tgtRGB, mix=0.6):
        h, w, c = im.shape
        res = im.copy()
        mask = np.zeros((h, w))
        mask[seg == 17] = 255
        mask = cv2.blur(mask, (5, 5))

        h_, s_, v_ = self.getAve(mask, im)
        print('ave v: ', h_, s_, v_)

        for row in range(0, h):
            for col in range(0, w):
                if mask[row, col] > 0 and (seg[row, col] == 17):
                    srcRGB = [res[row, col, 0], res[row, col, 1], res[row, col, 2]]
                    # srcRGB = [result[row,col,0],result[row,col,1],result[row,col,2]]
                    srcHSV = self.rgb2hsv(srcRGB[0], srcRGB[1], srcRGB[2])

                    tgtHSV = self.rgb2hsv(tgtRGB[0], tgtRGB[1], tgtRGB[2])
                    # ratio = 0.5/(tgtHSV[2] + 0.01)
                    # srcHSV[2] = math.pow(srcHSV[2],ratio)
                    #                 if srcHSV[2] < 0.01:
                    #                     srcHSV[2] += 0.05
                    if v_ < 0.1:
                        srcHSV[2] += 0.3
                        if srcHSV[2] > 1.0:
                            srcHSV[2] = 0.99
                    srcHSVnew = [tgtHSV[0], tgtHSV[1], srcHSV[2]]  # 只替换h,s到目标颜色
                    srcRGBnew = self.hsv2rgb(srcHSVnew[0], srcHSVnew[1], srcHSVnew[2])
                    resRGB = (1 - mix) * np.array(srcRGB) + mix * np.array(srcRGBnew)
                    res[row, col, 0], res[row, col, 1], res[row, col, 2] = int(resRGB[0]), int(resRGB[1]), int(
                        resRGB[2])
                # res[row,col,0],res[row,col,1],res[row,col,2] = hsv2rgb(srcHSVnew[0],srcHSVnew[1],srcHSVnew[2])
        return res, mask

    def change_eyebrow(self, im, seg, tgtRGB, mix=0.6):
        h, w, c = im.shape
        res = im.copy()
        mask = np.zeros((h, w))
        mask[seg == 2] = 255
        mask[seg == 3] = 255
        mask = cv2.blur(mask, (5, 5))

        for row in range(0, h):
            for col in range(0, w):
                if mask[row, col] > 0 and (seg[row, col] == 2 or seg[row, col] == 3):
                    # print(mask[row,col])
                    newmix = mix * mask[row, col] / 255.0
                    srcRGB = [res[row, col, 0], res[row, col, 1], res[row, col, 2]]
                    # srcRGB = [result[row,col,0],result[row,col,1],result[row,col,2]]
                    srcHSV = self.rgb2hsv(srcRGB[0], srcRGB[1], srcRGB[2])

                    tgtHSV = self.rgb2hsv(tgtRGB[0], tgtRGB[1], tgtRGB[2])
                    # ratio = 0.5/(tgtHSV[2] + 0.01)
                    # srcHSV[2] = math.pow(srcHSV[2],ratio)
                    #                 if srcHSV[2] < 0.01:
                    #                     srcHSV[2] += 0.05
                    # srcHSV[2] += 0.03
                    srcHSVnew = [tgtHSV[0], tgtHSV[1], srcHSV[2]]  # 只替换h,s到目标颜色
                    srcRGBnew = self.hsv2rgb(srcHSVnew[0], srcHSVnew[1], srcHSVnew[2])
                    resRGB = (1 - newmix) * np.array(srcRGB) + newmix * np.array(srcRGBnew)
                    # print(res[row,col,0])
                    res[row, col, 0], res[row, col, 1], res[row, col, 2] = int(resRGB[0]), int(resRGB[1]), int(
                        resRGB[2])
                # print('after: ',res[row,col,0])
                # res[row,col,0],res[row,col,1],res[row,col,2] = hsv2rgb(srcHSVnew[0],srcHSVnew[1],srcHSVnew[2])
        return res, mask

    def change_mouth(self, im, seg, tgtRGB, mix=0.6):
        h, w, c = im.shape
        res = im.copy()
        mask = np.zeros((h, w))
        mask[seg == 12] = 255
        mask[seg == 13] = 255
        mask = cv2.blur(mask, (5, 5))

        for row in range(0, h):
            for col in range(0, w):
                if mask[row, col] > 0 and (seg[row, col] == 12 or seg[row, col] == 13):
                    # print(mask[row,col])
                    newmix = mix * mask[row, col] / 255.0
                    srcRGB = [res[row, col, 0], res[row, col, 1], res[row, col, 2]]
                    # srcRGB = [result[row,col,0],result[row,col,1],result[row,col,2]]
                    srcHSV = self.rgb2hsv(srcRGB[0], srcRGB[1], srcRGB[2])

                    tgtHSV = self.rgb2hsv(tgtRGB[0], tgtRGB[1], tgtRGB[2])
                    # ratio = 0.5/(tgtHSV[2] + 0.01)
                    # srcHSV[2] = math.pow(srcHSV[2],ratio)
                    #                 if srcHSV[2] < 0.01:
                    #                     srcHSV[2] += 0.05
                    # srcHSV[2] += 0.03
                    srcHSVnew = [tgtHSV[0], tgtHSV[1], srcHSV[2]]  # 只替换h,s到目标颜色
                    srcRGBnew = self.hsv2rgb(srcHSVnew[0], srcHSVnew[1], srcHSVnew[2])
                    resRGB = (1 - newmix) * np.array(srcRGB) + newmix * np.array(srcRGBnew)
                    # print(res[row,col,0])
                    res[row, col, 0], res[row, col, 1], res[row, col, 2] = int(resRGB[0]), int(resRGB[1]), int(
                        resRGB[2])
                # res[row,col,0],res[row,col,1],res[row,col,2] = hsv2rgb(srcHSVnew[0],srcHSVnew[1],srcHSVnew[2])
        return res, mask

    def Change(self, im, name, tgtRGB, mix=0.6):
        seg = self.segmentation(im)
        if name == 'hair':
            res, mask = self.change_hair(im, seg, tgtRGB, mix)
        if name == 'mouth':
            res, mask = self.change_mouth(im, seg, tgtRGB, mix)
        if name == 'eyebrow':
            res, mask = self.change_eyebrow(im, seg, tgtRGB, mix)
        return res, mask

    def saveImg(self, name, res):
        # plt.imsave(name, res)
        return


if __name__ == "__main__":
    cgclr = changeFace()
    # img = Image.open('G:/Deecamp/pic2.png')
    # lr_img = np.array(img)
    # img = cv2.imread('G:/Deecamp/pic2.png')[:, :, ::-1]
    img = np.array(Image.open('G:/Deecamp/pic2.png'))
    # cv2.imwrite('G:/Deecamp/rgb.png', img)
    imsave('G:/Deecamp/rgb.png', img)
    res, mask = cgclr.change2(img, 'mouth', [100, 0, 0])
    # cgclr.getMask(img, 'skin')
    # cv2.imwrite('G:/Deecamp/changeMouth.png', res)
    imsave('G:/Deecamp/changeMouth.png', res)
    print('successful')
