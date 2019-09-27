
# -*- coding:utf8 -*-
from flask import Flask, request, json
# import sql
import base64
import numpy as np
import cv2
import demo2
from FaceDetector.FaceDetectorPro import faceDetect_server
from ISR.models import RDN, RRDN
from PIL import Image
from change_Styclass import changestyle
# import matplotlib.pyplot as plt
#
# from Flask.CARN_pytorch.carn.cap import CARN_SR
# from Flask.segmodels.parser import face_parser
# import data_helper as dt
from changeClass import changeFace

app = Flask(__name__)
# 全局化
# carn = CARN_SR()
cgclr = changeFace()
stgan = demo2.STGAN()
rdn = RDN(arch_params={'C': 6, 'D': 20, 'G': 64, 'G0': 64, 'x': 2})
rdn.model.load_weights(
    'G:/Deecamp/STGAN_ZW/STGAN/ImageSuperResolution/ArtefactCancelling/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf')

img = Image.open('G:/Deecamp/STGAN_ZW/STGAN/ImageSuperResolution/image/20121.jpg')

lr_img = np.array(img)
rdn.predict(lr_img)
cgclr.getMask(lr_img, 'skin')

cgstyle = changestyle()
im = cv2.imread("G:/Deecamp/STGAN_ZW/STGAN/transform_folder/fj2.png")
body = cgstyle.inference(im, 1)


@app.route('/')
def hello_world():
    return "Hello"


@app.route('/changeColor', methods=['POST', 'GET'])
# 可能需要用两个函数来处理图片
def changeColor():
    if request.method == 'POST':
        dic = request.get_json()
        pic = dic['file']
        RED = int(dic['RED'])
        GREEN = int(dic['GREEN'])
        BLUE = int(dic['BLUE'])
        mix = int(dic['mix'])

        # print(pic)
        # sql存储照片
        # sql.store(pic)

        # 图片以base64编码，在此解压
        pic = base64.b64decode(pic)
        pic = np.frombuffer(pic, np.uint8)
        pic = cv2.imdecode(pic, cv2.IMREAD_COLOR)[..., ::-1]

        # pic现在是numpy array，调用接口进行处理
        type = dic['type']
        tgtRGB = [RED, GREEN, BLUE]
        pic, mask = cgclr.Change(pic, type, tgtRGB, mix / 10.0)

        pic = pic[..., ::-1]
        # pic转base64编码
        retval, buffer = cv2.imencode('.jpg', pic)
        pic = base64.b64encode(buffer)
        pic = pic.decode()
        dic = {'res_data': pic}
        return json.dumps(dic)


@app.route('/changeAttr', methods=['POST', 'GET'])
def changeAttr():
    if request.method == 'POST':
        dic = request.get_json()
        pic = dic['file']
        attr = [dic['attr']]

        # print(pic)
        # sql存储照片
        # sql.store(pic)

        # 图片以base64编码，在此解压
        pic = base64.b64decode(pic)
        pic = np.frombuffer(pic, np.uint8)
        pic = cv2.imdecode(pic, cv2.IMREAD_COLOR)[..., ::-1]

        pic = cv2.resize(pic, (128, 128))
        # print("PIC", pic.shape, type(pic), type(pic[0][0][0]))

        # mask = cgclr.getMask(lr_img,'skin')
        # mask = mask/255
        # mask = np.reshape(mask,(256,256,1))
        pic_1 = cv2.resize(pic, (128, 128))  # resize为128*128

        # pic现在是numpy array，调用接口进行处理
        labels = stgan.classifier(pic_1)
        # stgan.verification(path_list, labels, name_list)
        pic_GEN = stgan.inference(pic_1, labels, attr)[0]
        pic_1 = rdn.predict(pic_GEN)
        # print(pic.shape, type(pic[0][0][0]))
        # pic_1 = carn.inference(pic_GEN)
        # mask_ori = (mask + 1) % 2
        # pic = pic * mask_ori + pic_1 *mask
        pic = pic_1[..., ::-1]
        # pic转base64编码
        retval, buffer = cv2.imencode('.jpg', pic)
        pic = base64.b64encode(buffer)
        pic = pic.decode()
        dic = {'res_data': pic}
        return json.dumps(dic)


@app.route("/changeStyle", methods=['POST', 'GET'])
def changeStyle():
    if request.method == 'POST':
        dic = request.get_json()
        pic = dic['file']
        type = int(dic['type'])
        # print(pic)
        # sql存储照片
        # sql.store(pic)

        # 图片以base64编码，在此解压
        pic = base64.b64decode(pic)
        pic = np.frombuffer(pic, np.uint8)
        pic = cv2.imdecode(pic, cv2.IMREAD_COLOR)
        print("sss")
        # pic现在是numpy array，调用接口进行处理

        pic = cgstyle.inference(pic, type)  ############################

        print("ccc")
        # pic转base64编码
        retval, buffer = cv2.imencode('.jpg', pic)
        pic = base64.b64encode(buffer)
        pic = pic.decode()
        dic = {'res_data': pic}
        return json.dumps(dic)


@app.route('/changeSize', methods=['POST', 'GET'])
def changeSize():
    if request.method == 'POST':
        dic = request.get_json()
        pic = dic['file']

        # print(pic)
        # sql存储照片
        # sql.store(pic)

        # 图片以base64编码，在此解压
        pic = base64.b64decode(pic)
        pic = np.frombuffer(pic, np.uint8)
        pic = cv2.imdecode(pic, cv2.IMREAD_COLOR)

        # pic现在是numpy array，调用接口进行处理

        pic = faceDetect_server(pic, imgSize=(512, 512))
        pic = rdn.predict(pic)  # 超分为512*512

        # pic转base64编码
        retval, buffer = cv2.imencode('.jpg', pic)
        pic = base64.b64encode(buffer)
        pic = pic.decode()
        dic = {'res_data': pic}
        return json.dumps(dic)


if __name__ == '__main__':
    app.run(host='114.212.173.176', port=5050)

    # tgtRGB = [255,0,0]
    # changePart = 'mouth'
    # mix = 0.6
    # res, mask = cgclr.change2(lr_img, changePart, tgtRGB, mix)
    # #res,mask = cgclr.Change(lr_img,changePart,tgtRGB,mix)
    # # print(res.shape)
    # print("lr_img", lr_img.shape, lr_img.dtype)
    # mask = cgclr.getMask(lr_img, 'skin')
    # mask = mask / 255
    # mask = np.reshape(mask,(128, 128, 1))
    # pic_1 = cv2.resize(lr_img, (128, 128))  # resize为128*128
    #
    # # pic现在是numpy array，调用接口进行处理
    # labels = stgan.classifier(pic_1)
    # # stgan.verification(path_list, labels, name_list)
    # pic_GEN = stgan.inference(pic_1, labels, ['Pale_Skin'])[0][..., ::-1]
    # # print(pic.shape, type(pic[0][0][0]))
    # # pic_1 = rdn.predict(pic_GEN)
    # mask_ori = (mask + 1) % 2
    # pic = lr_img * mask_ori + pic_GEN * mask
    #
    # cv2.imshow('img',pic)
    # cv2.waitKey()
    # carn = CARN_SR()
    # pic_1 = carn.inference(lr_img)
    # cv2.imshow('a',pic_1)
