from flask import Flask,request
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from matplotlib import pyplot as plt
from imageio import imread, imsave
import tensorflow as tf
import os
import glob
import argparse
from transform_folder.models.parser import face_parser
#import face_recognition   ###########################
from PIL import Image
import cv2 as cv
import argparse

class changestyle(object):
    def __init__(self):
        self.g_x=tf.Graph()
        self.sess_b=tf.Session(graph=self.g_x)
        
        self.batch_size = 1
        self.img_size = 256
        x,y,xs=self.init_models()
        self.X=x
        self.Y=y
        self.Xs=xs
        self.prs=face_parser.FaceParser()

    def init_models(self):
        with self.sess_b.as_default():
            with self.sess_b.graph.as_default():
#                 saver = tf.train.import_meta_graph(os.path.join('./model', 'model.meta'))
                saver = tf.train.import_meta_graph(os.path.join('/data/code/STGAN_latest/STGAN/transform_folder/model', 'model.meta'))
                saver.restore(self.sess_b,
                              tf.train.latest_checkpoint('/data/code/STGAN_latest/STGAN/transform_folder/model'))
                X = self.sess_b.graph.get_tensor_by_name('X:0')
                Y = self.sess_b.graph.get_tensor_by_name('Y:0')
                Xs = self.sess_b.graph.get_tensor_by_name('generator/xs:0')
                # input = tf.placeholder(shape=(None, 128, 128, 3), dtype=tf.float32)
                # logits = att_models.classifier(input, reuse=False, training=False)
                # saver = tf.train.Saver()
                # saver.restore(self.sess_b, '/home/zou/deeplearning/GAN/STGAN/att_classification/checkpoints/128.ckpt')
        return X, Y, Xs

    def preprocess(self,img):
        return (img / 255. - 0.5) * 2

    def deprocess(self,img):
        return (img + 1) / 2

    def inference(self,im,type):
        # # # 定义你的带转换图像地址
        # im_url = request.args.get("url")
        # 定义你的目标风格图像地址

        if type == 0:
            makeup_photo = "/data/code/STGAN_latest/STGAN/transform_folder/imgs/makeup/gete.png"
        elif type == 1:
            makeup_photo = "/data/code/STGAN_latest/STGAN/transform_folder/imgs/makeup/vFG112.png"
        elif type == 2:
            makeup_photo = "/data/code/STGAN_latest/STGAN/transform_folder/imgs/makeup/vFG137.png"
        elif type == 3:
            makeup_photo = "/data/code/STGAN_latest/STGAN/transform_folder/imgs/makeup/vFG756.png"
        elif type == 4:
            makeup_photo = "/data/code/STGAN_latest/STGAN/transform_folder/imgs/makeup/vRX916.png"
        elif type == 5:
            makeup_photo = "/data/code/STGAN_latest/STGAN/transform_folder/imgs/makeup/XMY-014.png"
        elif type == 6:
            makeup_photo = "/data/code/STGAN_latest/STGAN/transform_folder/imgs/makeup/XMY-074.png"
        else:
            makeup_photo = "/data/code/STGAN_latest/STGAN/transform_folder/imgs/makeup/XMY-136.png"

        """人脸截取"""
        # 此处将输入的图片通过face_recognition
        # image = face_recognition.load_image_file("D:\\daima\\gan\\BeautyGAN-master\\transform\\imgs\\no_makeup\\fj2.png")
        #
        # face_locations = face_recognition.face_locations(image)

        # image = cv.imread(im_url)
        # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # image = Image.fromarray(im)
        image = im[:, :, ::-1]
        detector = cv.CascadeClassifier("/data/code/STGAN_latest/STGAN/transform_folder/haarcascade_frontalface_alt.xml")
        rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=2, minSize=(10, 10),
                                          flags=cv.CASCADE_SCALE_IMAGE)  # (x,y,w,h)
        x, y, w, h = rects[0]
        y = y - 30
        h = h + 60
        # print(rects )
        # roiImg = image[y:(y+h),x:(x+w)]
        roiImg = image[y:(y + h), x:(x + w)]
        # imsave('result_face.jpg', roiImg)

        """风格迁移"""
        no_makeup = cv.resize(roiImg, (self.img_size, self.img_size))
        X_img = np.expand_dims(self.preprocess(no_makeup), 0)
        makeups = glob.glob(os.path.join('imgs', 'makeup', '*.*'))
        result = np.ones((2 * self.img_size, (len(makeups) + 1) * self.img_size, 3))
        result[self.img_size: 2 * self.img_size, :self.img_size] = no_makeup / 255.

        makeup = cv.resize(imread(makeup_photo), (self.img_size, self.img_size))
        Y_img = np.expand_dims(self.preprocess(makeup), 0)
        Xs_ = self.sess_b.run(self.Xs, feed_dict={self.X: X_img, self.Y: Y_img})
        result = cv.resize(Xs_[0], (w, h))
        result = (result * 0.5 + 0.5) * 255
        im = result


        # im = resize_image(im) # Resize image to prevent GPU OOM.
        im = cv.resize(im, (w, h))
        h, w, _ = im.shape
        #out = self.prs.parse_face(im)
        out = self.prs.parse_face(im)
        # print(type(out[0]))
        # imsave('result1.jpg', out[0])
        # print(out[0].shape)
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
        print("mask_shape", mask.shape)

        # # print(out[0][150,150])
        face = im * mask
        # imsave('./resultseg.jpg', face[:, :, ::-1])
        # print(face.shape)
        face = face[:, :, ::-1]
        body = image[:, :, ::-1]
        body[y:(y + h), x:(x + w)][mask] = face[mask]
        body = np.uint8(body)
#         imsave('./transformfj.png', body)
        return body

if __name__ == '__main__':
    cgstyle = changestyle()
    im=cv.imread("./fj2.png")
    #im = im[:,:,::-1]
    body=cgstyle.inference(im,type = 0)
    
    






    #
    # stgan = STGAN()
    # # path_list, name_list = stgan.crop(img_size=256)
    # img = cv2.imread('/home/zou/deeplearning/GAN/STGAN/test/crop/0.jpg')
    # labels = stgan.classifier(img)
    # # stgan.verification(path_list, labels, name_list)
    # result = stgan.inference(img, labels, ["Blond_Hair", "Goatee"])
    # print(result)
    # cv2.imwrite('./test/result/0.jpg', result[0])
    # # plt.show()
    # # cv2.waitKey()




# class changeFace():
#     def __init__(self):
#         # self.im = im
#         # self.change_name = change_name
#         # self.tgtRGB = tgtRGB
#         # self.mix = mix
#         self.prs = face_parser.FaceParser()