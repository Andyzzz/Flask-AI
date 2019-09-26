import math
import warnings

warnings.filterwarnings("ignore")
import cv2
import numpy as np
#from matplotlib import pyplot as plt
import sys
#sys.path.append('C:\\Users\\Administrator\\Downloads\\face_toolbox_keras\\')
from models.parser import face_parser
#from utils.visualize import show_parsing_with_annos


# 改变图片尺寸

def resize_image(im, max_size=768):
    if np.max(im.shape) > max_size:
        ratio = max_size / np.max(im.shape)
        print(f"Resize image to ({str(int(im.shape[1] * ratio))}, {str(int(im.shape[0] * ratio))}).")
        return cv2.resize(im, (0, 0), fx=ratio, fy=ratio)
    return im


def change(im, seg, changePart, tgtRGB):
    segDict = {'background': [0], 'skin': [1], 'left eyebrow': [2], 'right eyebrow': [3], 'left eye': [4],
               'right eye': [5], 'glasses': [6],
               'left ear': [7], 'right ear': [8], 'earings': [9], 'nose': [10], 'mouth': [12, 13], 'upper lip': [12],
               'lower lip': [13], 'neck': [14],
               'neck_l': [15], 'cloth': [16], 'hair': [17], 'hat': [18], 'eyes': [4, 5], 'ears': [7, 8],
               'eyebrows': [2, 3]
               }

    h, w, c = im.shape
    res = im.copy()

    for row in range(0, h):
        for col in range(0, w):
            if seg[row, col] in segDict[changePart]:
                srcRGB = [res[row, col, 0], res[row, col, 1], res[row, col, 2]]
                # srcRGB = [result[row,col,0],result[row,col,1],result[row,col,2]]
                srcHSV = rgb2hsv(srcRGB[0], srcRGB[1], srcRGB[2])

                tgtHSV = rgb2hsv(tgtRGB[0], tgtRGB[1], tgtRGB[2])

                srcHSVnew = [tgtHSV[0], tgtHSV[1], srcHSV[2]]  # 只替换h,s到目标颜色
                res[row, col, 0], res[row, col, 1], res[row, col, 2] = hsv2rgb(srcHSVnew[0], srcHSVnew[1], srcHSVnew[2])
    return res


# hsv变RGB
def hsv2rgb(h, s, v):
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


# RGB转HSV
def rgb2hsv(r, g, b):
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


def change2(im, seg, changePart, tgtRGB, mix=0.5):
    segDict = {'background': [0], 'skin': [1], 'left eyebrow': [2], 'right eyebrow': [3], 'left eye': [4],
               'right eye': [5], 'glasses': [6],
               'left ear': [7], 'right ear': [8], 'earings': [9], 'nose': [10], 'mouth': [12, 13], 'upper lip': [12],
               'lower lip': [13], 'neck': [14],
               'neck_l': [15], 'cloth': [16], 'hair': [17], 'hat': [18], 'eyes': [4, 5], 'ears': [7, 8],
               'eyebrows': [2, 3]
               }

    h, w, c = im.shape
    res = im.copy()

    for row in range(0, h):
        for col in range(0, w):
            if seg[row, col] in segDict[changePart]:
                srcRGB = [res[row, col, 0], res[row, col, 1], res[row, col, 2]]
                # srcRGB = [result[row,col,0],result[row,col,1],result[row,col,2]]
                srcHSV = rgb2hsv(srcRGB[0], srcRGB[1], srcRGB[2])

                tgtHSV = rgb2hsv(tgtRGB[0], tgtRGB[1], tgtRGB[2])
                # ratio = 0.5/(tgtHSV[2] + 0.01)
                # srcHSV[2] = math.pow(srcHSV[2],ratio)
                #                 if srcHSV[2] < 0.01:
                #                     srcHSV[2] += 0.05
                srcHSV[2] += 0.03
                srcHSVnew = [tgtHSV[0], tgtHSV[1], srcHSV[2]]  # 只替换h,s到目标颜色
                srcRGBnew = hsv2rgb(srcHSVnew[0], srcHSVnew[1], srcHSVnew[2])
                resRGB = (1 - mix) * np.array(srcRGB) + mix * np.array(srcRGBnew)
                res[row, col, 0], res[row, col, 1], res[row, col, 2] = int(resRGB[0]), int(resRGB[1]), int(resRGB[2])
                # res[row,col,0],res[row,col,1],res[row,col,2] = hsv2rgb(srcHSVnew[0],srcHSVnew[1],srcHSVnew[2])
    return res





def change_eyebrow(im, seg, tgtRGB, mix=0.5):
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
                srcHSV = rgb2hsv(srcRGB[0], srcRGB[1], srcRGB[2])

                tgtHSV = rgb2hsv(tgtRGB[0], tgtRGB[1], tgtRGB[2])
                # ratio = 0.5/(tgtHSV[2] + 0.01)
                # srcHSV[2] = math.pow(srcHSV[2],ratio)
                #                 if srcHSV[2] < 0.01:
                #                     srcHSV[2] += 0.05
                # srcHSV[2] += 0.03
                srcHSVnew = [tgtHSV[0], tgtHSV[1], srcHSV[2]]  # 只替换h,s到目标颜色
                srcRGBnew = hsv2rgb(srcHSVnew[0], srcHSVnew[1], srcHSVnew[2])
                resRGB = (1 - newmix) * np.array(srcRGB) + newmix * np.array(srcRGBnew)
                # print(res[row,col,0])
                res[row, col, 0], res[row, col, 1], res[row, col, 2] = int(resRGB[0]), int(resRGB[1]), int(resRGB[2])
                # print('after: ',res[row,col,0])
                # res[row,col,0],res[row,col,1],res[row,col,2] = hsv2rgb(srcHSVnew[0],srcHSVnew[1],srcHSVnew[2])
    return res, mask


def change_mouth(im, seg, tgtRGB, mix=0.5):
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
                srcHSV = rgb2hsv(srcRGB[0], srcRGB[1], srcRGB[2])

                tgtHSV = rgb2hsv(tgtRGB[0], tgtRGB[1], tgtRGB[2])
                # ratio = 0.5/(tgtHSV[2] + 0.01)
                # srcHSV[2] = math.pow(srcHSV[2],ratio)
                #                 if srcHSV[2] < 0.01:
                #                     srcHSV[2] += 0.05
                # srcHSV[2] += 0.03
                srcHSVnew = [tgtHSV[0], tgtHSV[1], srcHSV[2]]  # 只替换h,s到目标颜色
                srcRGBnew = hsv2rgb(srcHSVnew[0], srcHSVnew[1], srcHSVnew[2])
                resRGB = (1 - newmix) * np.array(srcRGB) + newmix * np.array(srcRGBnew)
                # print(res[row,col,0])
                res[row, col, 0], res[row, col, 1], res[row, col, 2] = int(resRGB[0]), int(resRGB[1]), int(resRGB[2])
                # print('after: ',res[row,col,0])
                # res[row,col,0],res[row,col,1],res[row,col,2] = hsv2rgb(srcHSVnew[0],srcHSVnew[1],srcHSVnew[2])
    return res, mask



def pre(im):

    prs = face_parser.FaceParser()
    out = prs.parse_face(im)
    h,w= out[0].shape


    seg = out[0]

    # 嘴巴
    tgtRGB = [255,0,0]
    res,mask = change_mouth(im,seg,tgtRGB,1)

    # 眉毛
    # tgtRGB = [139,69,19]
    # seg = out[0]
    # res,mask = change_eyebrow(im,seg,tgtRGB,1)
    #plt.imsave('C:\\Users\\Administrator\\Desktop\\result.jpg',res)
    cv2.imwrite('C:\\Users\\Administrator\\Desktop\\result.jpg',res[...,::-1])
    return res

