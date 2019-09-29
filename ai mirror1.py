from flask import Flask, request, json
import base64
import numpy as np
import cv2
from ISR.models import RDN, RRDN
from PIL import Image
import matplotlib.pyplot as plt
import data_helper as dt
# import sql

# sys.path.append("/data/code/STGAN_latest/STGAN/")
import demo2
from FaceDetector.FaceDetectorPro import faceDetect_server
from change_Styclass import changestyle
# from segmodels.parser import face_parser
from changeClass import changeFace
import faceswapOnline as fs
import sys

sys.path.append("G:/Deecamp/STGAN_ZW/STGAN/")
app = Flask(__name__)

# 小程序里所有的妆容变换的示例图标
with open('G:/Deecamp/STGAN_ZW/STGAN/makeup/0.png', 'rb') as f:
    a1 = base64.b64encode(f.read()).decode()
with open('G:/Deecamp/STGAN_ZW/STGAN/makeup/1.png', 'rb') as f:
    a2 = base64.b64encode(f.read()).decode()
with open('G:/Deecamp/STGAN_ZW/STGAN/makeup/2.png', 'rb') as f:
    a3 = base64.b64encode(f.read()).decode()
with open('G:/Deecamp/STGAN_ZW/STGAN/makeup/3.png', 'rb') as f:
    a4 = base64.b64encode(f.read()).decode()
with open('G:/Deecamp/STGAN_ZW/STGAN/makeup/4.png', 'rb') as f:
    a5 = base64.b64encode(f.read()).decode()
with open('G:/Deecamp/STGAN_ZW/STGAN/makeup/5.png', 'rb') as f:
    a6 = base64.b64encode(f.read()).decode()
with open('G:/Deecamp/STGAN_ZW/STGAN/makeup/6.png', 'rb') as f:
    a7 = base64.b64encode(f.read()).decode()
with open('G:/Deecamp/STGAN_ZW/STGAN/makeup/7.png', 'rb') as f:
    a8 = base64.b64encode(f.read()).decode()

with open('history.jpg', 'rb') as f:
    history = base64.b64encode(f.read()).decode()
with open('marval.jpg', 'rb') as f:
    marval = base64.b64encode(f.read()).decode()
with open('springFastival.jpg', 'rb') as f:
    springFestival = base64.b64encode(f.read()).decode()

# 全局化
cgclr = changeFace()
stgan = demo2.STGAN()
rdn = RDN(arch_params={'C': 6, 'D': 20, 'G': 64, 'G0': 64, 'x': 2})

rdn.model.load_weights(
    'G:/Deecamp/STGAN_ZW/STGAN/ImageSuperResolution/ArtefactCancelling/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf')

img = Image.open('G:/Deecamp/STGAN_ZW/STGAN/ImageSuperResolution/image/20121.jpg')
lr_img = np.array(img)
rdn.predict(lr_img)
cgclr.getMask(lr_img, 'skin')
# print('successful!')

cgstyle = changestyle()
im = cv2.imread("G:/Deecamp/STGAN_ZW/STGAN/transform_folder/fj2.png")
body = cgstyle.inference(im, 1)


@app.route('/Hello')
def hello_world():
    return "Hello"


@app.route('/init', methods=['POST', 'GET'])
def init():
    # 把makeup的图片传过去
    # 把gif也传过去
    if request.method == 'GET':
        # 编码图片
        dic = {}
        dic['a1'] = a1
        dic['a2'] = a2
        dic['a3'] = a3
        dic['a4'] = a4
        dic['a5'] = a5
        dic['a6'] = a6
        dic['a7'] = a7
        dic['a8'] = a8

        return json.dumps(dic)


#######xu
@app.route('/story_init', methods=['POST', 'GET'])
def story_init():
    # story的图片传过去
    if request.method == 'GET':
        # 编码图片
        dic = {}
        dic['a1'] = history
        dic['a2'] = marval
        dic['a3'] = springFestival
        return json.dumps(dic)


@app.route('/story_init_1', methods=['POST', 'GET'])
def story_init_1():
    # story的图片传过去
    if request.method == 'POST':
        dic = request.get_json()
        pic = dic['file']
        story = dic['type']

        # # print(pic)
        # sql存储照片
        # sql.store(pic)

        # 图片以base64编码，在此解压
        pic = base64.b64decode(pic)
        pic = np.frombuffer(pic, np.uint8)
        pic = cv2.imdecode(pic, cv2.IMREAD_COLOR)

        # pic现在是numpy array，调用接口进行处理
        a = fs.Faceswap()
        if story == 'marvel':
            path = "G:/Deecamp/STGAN_ZW/STGAN/face_swap/marWithTitle"
        elif story == "history":
            path = "G:/Deecamp/STGAN_ZW/STGAN/face_swap/history"
        elif story == "SpringFestival":
            path = "G:/Deecamp/STGAN_ZW/STGAN/face_swap/SpringFestival"
        else:
            path = None
            # print("error in ai mirror path")
        # path = 'C:/Users/rht/PycharmProjects/face_swap/mar'
        outputs, subtitles = a.test_online(pic, path)
        dic = {}
        for i, output in enumerate(outputs):
            # pic转base64编码
            retval, buffer = cv2.imencode('.jpg', output)
            pic = base64.b64encode(buffer)
            pic = pic.decode()
            dict_pic = {'pic': pic,
                        'subtitle': subtitles[i]
                        }
            dic[i] = dict_pic
            # dic = {'res_data'+str(i): pic}
        return json.dumps(dic)


@app.route('/changeColor', methods=['POST', 'GET'])
# 这里的methods的意思
# 可能需要用两个函数来处理图片
def changeColor():
    if request.method == 'POST':
        dic = request.get_json()
        pic = dic['file']
        RED = int(dic['RED'])
        GREEN = int(dic['GREEN'])
        BLUE = int(dic['BLUE'])
        mix = int(dic['mix'])

        # sql存储照片
        # sql.store(pic)

        # 图片以base64编码，在此解压
        pic = base64.b64decode(pic)
        pic = np.frombuffer(pic, np.uint8)
        pic = cv2.imdecode(pic, cv2.IMREAD_COLOR)[..., ::-1]

        # pic现在是numpy array，调用接口进行处理
        type = dic['type']
        tgtRGB = [RED, GREEN, BLUE]
        pic, mask = cgclr.change2(pic, type, tgtRGB, mix / 10.0)

        pic = pic[..., ::-1]
        # pic转base64编码
        retval, buffer = cv2.imencode('.jpg', pic)
        pic = base64.b64encode(buffer)
        pic = pic.decode()
        dic = {'res_data': pic}
        return json.dumps(dic)


# 想看传入json的图片格式，以及传入前端后怎么如何变成图片
@app.route("/changeStyle", methods=['POST', 'GET'])
def changeStyle():
    if request.method == 'POST':
        dic = request.get_json()
        pic = dic['file']
        type = int(dic['type'])
        # # print(pic)
        # sql存储照片
        # sql.store(pic)

        # 图片以base64编码，在此解压
        pic = base64.b64decode(pic)
        pic = np.frombuffer(pic, np.uint8)
        pic = cv2.imdecode(pic, cv2.IMREAD_COLOR)
        # print("sss")
        # pic现在是numpy array，调用接口进行处理

        pic = cgstyle.inference(pic, type)

        # print("ccc")
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

        # 图片以base64编码，在此解压
        pic = base64.b64decode(pic)
        pic = np.frombuffer(pic, np.uint8)
        pic = cv2.imdecode(pic, cv2.IMREAD_COLOR)[..., ::-1]


        # mask = cgclr.getMask(lr_img,'skin')
        # mask = mask/255
        # mask = np.reshape(mask,(256,256,1))
        pic_1 = cv2.resize(pic, (128, 128))  # resize为128*128
        # 这里有问题，代码重复

        # pic现在是numpy array，调用接口进行处理
        labels = stgan.classifier(pic_1)
        # stgan.verification(path_list, labels, name_list)
        pic_GEN = stgan.inference(pic_1, labels, attr)[0][..., ::-1]
        # # print(pic.shape, type(pic[0][0][0]))
        #         pic_GEN = cv2.resize(pic_GEN, (256, 256))
        #         pic_1 = rdn.predict(pic_GEN)
        # rdn.predict(pic_GEN)需要传入的是BGR？
        # mask_ori = (mask + 1) % 2
        # pic = pic * mask_ori + pic_1 *mask
        pic = pic_GEN
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
        pic = dic['file']   # 从前端获得的图片

        # 图片以base64编码，在此解压
        pic = base64.b64decode(pic)
        pic = np.frombuffer(pic, np.uint8)
        pic = cv2.imdecode(pic, cv2.IMREAD_COLOR)
        # print('shape before changeSize: ', pic.shape)
        # pic现在是numpy array，调用接口进行处理
        cv2.imwrite('G:/Deecamp/pic.png', pic)
        pic = faceDetect_server(pic, imgSize=(128, 128))  # 128， 128
        cv2.imwrite('G:/Deecamp/pic2.png', pic)
        # print('shape after changeSize: ', pic.shape)
        # pic = rdn.predict(pic)  # 超分为256*256

        # pic转base64编码
        retval, buffer = cv2.imencode('.jpg', pic)
        pic = base64.b64encode(buffer)
        pic = pic.decode()
        # print('type pic ', type(pic))
        dic = {'res_data': pic}
        return json.dumps(dic)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)
    # app.run(host='114.212.173.176', port=8080)

    # tgtRGB = [255,0,0]
    # changePart = 'mouth'
    # mix = 0.6
    # res, mask = cgclr.change2(lr_img, changePart, tgtRGB, mix)
    # #res,mask = cgclr.Change(lr_img,changePart,tgtRGB,mix)
    # # # print(res.shape)
    # # print("lr_img", lr_img.shape, lr_img.dtype)
    # mask = cgclr.getMask(lr_img, 'skin')
    # mask = mask / 255
    # mask = np.reshape(mask,(128, 128, 1))
    # pic_1 = cv2.resize(lr_img, (128, 128))  # resize为128*128
    #
    # # pic现在是numpy array，调用接口进行处理
    # labels = stgan.classifier(pic_1)
    # # stgan.verification(path_list, labels, name_list)
    # pic_GEN = stgan.inference(pic_1, labels, ['Pale_Skin'])[0][..., ::-1]
    # # # print(pic.shape, type(pic[0][0][0]))
    # # pic_1 = rdn.predict(pic_GEN)
    # mask_ori = (mask + 1) % 2
    # pic = lr_img * mask_ori + pic_GEN * mask
    #
    # cv2.imshow('img',pic)
    # cv2.waitKey()
    # pic = lr_img
    # # print('下面出现错误')
    # pic = faceDetect_server(pic, imgSize=(128, 128))
    # pic = rdn.predict(pic)
    # # print('DEBUG>>>>> \n faceDetect_server is ok \n rdn is ok')
