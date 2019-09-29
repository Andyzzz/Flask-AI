from flask import Flask, request
import warnings
import numpy as np
from imageio import imread, imsave
import tensorflow as tf
import os
import glob
from segmodels.parser import face_parser
from PIL import Image
import cv2
warnings.filterwarnings("ignore")

class changestyle(object):
    def __init__(self):
        self.g_x = tf.Graph()
        self.sess_b = tf.Session(graph=self.g_x)

        self.batch_size = 1
        self.img_size = 256
        x, y, xs = self.init_models()
        self.X = x
        self.Y = y
        self.Xs = xs
        self.prs = face_parser.FaceParser()

    def init_models(self):
        with self.sess_b.as_default():
            with self.sess_b.graph.as_default():
                #                 saver = tf.train.import_meta_graph(os.path.join('./model', 'model.meta'))
                saver = tf.train.import_meta_graph(
                    os.path.join('G:/Deecamp/STGAN_ZW/STGAN/transform_folder/model', 'model.meta'))
                saver.restore(self.sess_b,
                              tf.train.latest_checkpoint('G:/Deecamp/STGAN_ZW/STGAN/transform_folder/model'))
                X = self.sess_b.graph.get_tensor_by_name('X:0')
                Y = self.sess_b.graph.get_tensor_by_name('Y:0')
                Xs = self.sess_b.graph.get_tensor_by_name('generator/xs:0')
                # input = tf.placeholder(shape=(None, 128, 128, 3), dtype=tf.float32)
                # logits = att_models.classifier(input, reuse=False, training=False)
                # saver = tf.train.Saver()
                # saver.restore(self.sess_b, '/home/zou/deeplearning/GAN/STGAN/att_classification/checkpoints/128.ckpt')
        return X, Y, Xs

    def preprocess(self, img):
        return (img / 255. - 0.5) * 2

    def deprocess(self, img):
        return (img + 1) / 2

    def inference(self, im, type):
        # 定义你的目标风格图像
        if type == 0:
            makeup_photo = "G:/Deecamp/STGAN_ZW/STGAN/transform_folder/imgs/makeup/gete.png"
        elif type == 1:
            makeup_photo = "G:/Deecamp/STGAN_ZW/STGAN/transform_folder/imgs/makeup/vFG112.png"
        elif type == 2:
            makeup_photo = "G:/Deecamp/STGAN_ZW/STGAN/transform_folder/imgs/makeup/vFG137.png"
        elif type == 3:
            makeup_photo = "G:/Deecamp/STGAN_ZW/STGAN/transform_folder/imgs/makeup/vFG756.png"
        elif type == 4:
            makeup_photo = "G:/Deecamp/STGAN_ZW/STGAN/transform_folder/imgs/makeup/vRX916.png"
        elif type == 5:
            makeup_photo = "G:/Deecamp/STGAN_ZW/STGAN/transform_folder/imgs/makeup/XMY-014.png"
        elif type == 6:
            makeup_photo = "G:/Deecamp/STGAN_ZW/STGAN/transform_folder/imgs/makeup/XMY-074.png"
        else:
            makeup_photo = "G:/Deecamp/STGAN_ZW/STGAN/transform_folder/imgs/makeup/XMY-136.png"

        """人脸截取"""
        # 此处将输入的图片通过人脸检测
        # 先把im从RGB转换为BGR，image是待转换图像
        image = im[:, :, ::-1]
        imsave('G:/Deecamp/image.png', image)
        # 检测人脸范围
        detector = cv2.CascadeClassifier("G:/Deecamp/STGAN_ZW/STGAN/transform_folder/haarcascade_frontalface_alt.xml")
        rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10)  # (x,y,w,h)  , minSize=(10, 10),flags=cv2.CASCADE_SCALE_IMAGE
        x, y, w, h = rects[0]
        y = y - 30
        h = h + 60
        roiImg = image[y:(y + h), x:(x + w)]  # 注意，这里有可能image本身的大小小于y+h和x+w，所以最终尺寸不一定是(w,h)，所以后面要重新记录一下(w,h)
        h, w, _ = roiImg.shape   # 更新h,w的值
        # imsave('G:/Deecamp/result_face.jpg', roiImg)

        # 风格迁移
        no_makeup = cv2.resize(roiImg, (self.img_size, self.img_size))  # 把roiImg转换成256x256尺寸，模型只接收256x256尺寸的输入
        # imsave('G:/Deecamp/no_makeup.jpg', no_makeup)
        X_img = np.expand_dims(self.preprocess(no_makeup), 0)
        makeups = glob.glob(os.path.join('imgs', 'makeup', '*.*'))
        result = np.ones((2 * self.img_size, (len(makeups) + 1) * self.img_size, 3))
        result[self.img_size: 2 * self.img_size, :self.img_size] = no_makeup / 255.0

        makeup = cv2.resize(imread(makeup_photo), (self.img_size, self.img_size))
        Y_img = np.expand_dims(self.preprocess(makeup), 0)
        Xs_ = self.sess_b.run(self.Xs, feed_dict={self.X: X_img, self.Y: Y_img})
        result = cv2.resize(Xs_[0], (w, h))  # 将256x256尺寸再resize回roiImg的尺寸
        result = (result * 0.5 + 0.5) * 255

        im = cv2.resize(result, (w, h))
        # h, w, _ = im.shape
        out = self.prs.parse_face(im)
        # mask = (out[0] == 1 or out[0] == 10 or out[0] == 12 or out[0] == 13 or out[0] == 5 or out[0] == 4)
        mask = out[0] == 1
        mask_nose = out[0] == 10
        mask_ulip = out[0] == 12
        mask_llip = out[0] == 13
        mask_leye = out[0] == 4
        mask_reye = out[0] == 5
        mask_lbrow = out[0] == 2
        mask_rbrow = out[0] == 3
        mask_hair = out[0] == 17
        mask_face = mask | mask_nose | mask_ulip | mask_llip | mask_leye | mask_reye | mask_lbrow | mask_rbrow
        mask_head = mask_face | mask_hair
        # print(mask.shape)
        mask = np.expand_dims(mask_face, -1)
        mask = np.concatenate([mask, mask, mask], 2)
        # print("mask_shape", mask.shape)

        face = im * mask
        face = face[:, :, ::-1]
        body = image[:, :, ::-1]

        # print('body shape ', body.shape)
        # print('mask shape ', mask.shape)
        # print('face shape ', face.shape)

        body[y:(y + h), x:(x + w)][mask] = face[mask]  # 将妆容变换后的区域（鼻子、嘴唇、眼睛、眉毛）的值赋值回原来的人脸图像
        body = np.uint8(body)  # 返回bgr格式的，后面调用会再转回rgb
        # body = body[:, :, ::-1]
        # imsave('G:/Deecamp/transform.png', body)
        return body


if __name__ == '__main__':
    cgstyle = changestyle()
    im = cv2.imread("G:/Deecamp/STGAN_ZW/STGAN/transform_folder/test0.jpg") # BGR格式
    body = cgstyle.inference(im, type=0)

    # image = im[:, :, ::-1]  # 从BGR转成了RGB
    #
    # imsave('G:/Deecamp/image.png', image)
    # # 检测人脸范围
    # detector = cv2.CascadeClassifier("G:/Deecamp/STGAN_ZW/STGAN/transform_folder/haarcascade_frontalface_alt.xml")
    # rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10)
    # print('main rects: ', len(rects))
