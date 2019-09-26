#encoding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from functools import partial
import tensorflow as tf
# import tensorflow as tf
import att_classification.models as att_models
import models
import cv2
import data
import numpy as np
import pylib
import tflib as tl
import matplotlib.image as mp
import FaceDetector.FaceDetectorPro as dalao
import matplotlib.pyplot as plt


useful_attrs = {
    # eyebrows
    "Arched_Eyebrows": 2.5,     # 拱眉
    "Bushy_Eyebrows": 1.0,      # 粗眉毛

    # hairstyle
    "Bald": 0.5,                # 光头
    "Blond_Hair": 3.0,          # 金发
    "Bangs": 2.0,               # 刘海
    "Brown_Hair": 3.0,          # 棕发
    "Gray_Hair": 0.5,           # 灰发
    "Receding_Hairline": 3.0,   # 发际线升高

    # beard
    "Goatee": 2.0,              # 山羊胡(下颌胡)
    "Mustache": 2.5,            # 八字胡(上唇胡)

    # face
    "Pale_Skin": 2.0,           # 苍白的脸
    "Rosy_Cheeks": 1.5,         # 粉红的脸

    # nose
   "Pointy_Nose": 2.0,          # 尖鼻

}



class STGAN(object):
    def __init__(self, ):
        self.att_dict = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2,
                         'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6,
                         'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10,
                         'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13,
                         'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16,
                         'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19,
                         'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22,
                         'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25,
                         'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28,
                         'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31,
                         'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34,
                         'Wearing_Hat': 35, 'Wearing_Lipstick': 36,
                         'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}
        self.att_default = ["5_o_Clock_Shadow",
                            "Arched_Eyebrows",
                            "Attractive",
                            "Bags_Under_Eyes",
                            "Bald",
                            "Bangs",
                            "Big_Lips",
                            "Big_Nose",
                            "Black_Hair",
                            "Blond_Hair",
                            "Blurry",
                            "Brown_Hair",
                            "Bushy_Eyebrows",
                            "Chubby",
                            "Double_Chin",
                            "Eyeglasses",
                            "Goatee",
                            "Gray_Hair",
                            "Heavy_Makeup",
                            "High_Cheekbones",
                            "Male",
                            "Mouth_Slightly_Open",
                            "Mustache",
                            "Narrow_Eyes",
                            "No_Beard",
                            "Oval_Face",
                            "Pale_Skin",
                            "Pointy_Nose",
                            "Receding_Hairline",
                            "Rosy_Cheeks",
                            "Sideburns",
                            "Smiling",
                            "Straight_Hair",
                            "Wavy_Hair",
                            "Wearing_Earrings",
                            "Wearing_Hat",
                            "Wearing_Lipstick",
                            "Wearing_Necklace",
                            "Wearing_Necktie",
                            "Young"]
        self.g2 = tf.Graph()
        self.g1 = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess_cls = tf.Session(graph=self.g1, config=config)
        self.sess_GAN = tf.Session(graph=self.g2, config=config)
        e, f, g, h = self.init_GAN_models()
        b, c = self.init_cls_models()
        self.input = b
        self.logits = c
        self.xa_sample = e
        self._b_sample = f
        self.raw_b_sample = g
        self.x_sample = h

    def init_cls_models(self):
        with self.sess_cls.as_default():
            with self.sess_cls.graph.as_default():
                input = tf.placeholder(shape=(None, 128, 128, 3), dtype=tf.float32)
                logits = att_models.classifier(input, reuse=False, training=False)
                saver = tf.train.Saver()
                saver.restore(self.sess_cls, '/data/code/STGAN_latest/STGAN/att_classification/checkpoints/128.ckpt')

        return input, logits

    def init_GAN_models(self):
        inject_layers = 4
        stu_inject_layers = 4
        enc_dim = 64
        dec_dim = 64
        stu_dim = 64
        enc_layers = 5
        dec_layers = 5
        stu_layers = 4
        multi_inputs = 1
        shortcut_layers = 4
        one_more_conv = 0
        stu_kernel_size = 3
        stu_norm = None
        stu_state = 'stu'
        img_size = 128
        n_att = len(self.att_default)
        label = 'diff'
        use_stu = True


        with self.sess_GAN.as_default():
            with self.sess_GAN.graph.as_default():
                Genc = partial(models.Genc, dim=enc_dim, n_layers=enc_layers, multi_inputs=multi_inputs)
                Gdec = partial(models.Gdec, dim=dec_dim, n_layers=dec_layers, shortcut_layers=shortcut_layers,
                               inject_layers=inject_layers, one_more_conv=one_more_conv)
                Gstu = partial(models.Gstu, dim=stu_dim, n_layers=stu_layers, inject_layers=stu_inject_layers,
                               kernel_size=stu_kernel_size, norm=stu_norm, pass_state=stu_state)

                # inputs
                xa_sample = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3])
                _b_sample = tf.placeholder(tf.float32, shape=[None, n_att])
                raw_b_sample = tf.placeholder(tf.float32, shape=[None, n_att])

                # sample
                test_label = _b_sample - raw_b_sample if label == 'diff' else _b_sample
                if use_stu:
                    x_sample = Gdec(Gstu(Genc(xa_sample, is_training=False),
                                         test_label, is_training=False), test_label, is_training=False)
                else:
                    x_sample = Gdec(Genc(xa_sample, is_training=False), test_label, is_training=False)
                saver = tf.train.Saver()
                saver.restore(self.sess_GAN, '/data/code/STGAN_latest/STGAN/output/128/checkpoints/Epoch_(100)_(947of947).ckpt')

        return xa_sample, _b_sample, raw_b_sample, x_sample

    def crop(self, img_path='./test/ori', save_path='./test/crop', img_size=128):
        name_list = os.listdir(img_path)
        path_list = []
        for i in range(len(name_list)):
            dalao.faceDetect(os.path.join(img_path, name_list[i]), os.path.join(save_path, name_list[i]),
                             imgSize=(img_size, img_size))
            path_list.append(os.path.join(save_path, name_list[i]))
        return path_list, name_list

    def classifier(self, img):
        pred = tf.cast(tf.round(tf.nn.sigmoid(self.logits)), tf.int64)

        pre = []

        if np.shape(img)[1] != 128:
            img = cv2.resize(img, (128, 128))
        img = img / 255.0 * 2.0 - 1.0
        img = np.reshape(img, newshape=(1, 128, 128, 3))
        pre.extend(self.sess_cls.run(pred, feed_dict={self.input: img}))

        print(pre)
        atts = []
        for item in self.att_default:
            atts.append(pre[0][self.att_dict[item]])

        return atts

    def inference(self, imgs, labels, attr_list):
        atts = attr_list
        att_val = [useful_attrs[att] for att in atts]
        img_size = 128
        n_slide = 10
        test_slide = False
        thres_int = 0.5

        imgs = imgs / 255.0 * 2.0 - 1.0

        xa_sample_ipt = np.reshape(imgs, newshape=(1, img_size, img_size, 3))
        a_sample_ipt = np.reshape(labels, newshape=(1, 40))
        b_sample_ipt_list = [a_sample_ipt.copy() for _ in range(n_slide if test_slide else 1)]
        print(b_sample_ipt_list)
        for a in atts:
            i = self.att_default.index(a)
            b_sample_ipt_list[-1][:, i] = 1 - b_sample_ipt_list[-1][:, i]
            b_sample_ipt_list[-1] = data.Celeba.check_attribute_conflict(b_sample_ipt_list[-1],
                                                                         self.att_default[i], self.att_default)
        raw_a_sample_ipt = a_sample_ipt.copy()
        raw_a_sample_ipt = (raw_a_sample_ipt * 2 - 1) * thres_int
        for i, b_sample_ipt in enumerate(b_sample_ipt_list):
            _b_sample_ipt = (b_sample_ipt * 2 - 1) * thres_int
            if not test_slide:
                if atts:  # i must be 0
                    for t_att, t_int in zip(atts, att_val):
                        _b_sample_ipt[..., atts.index(t_att)] = _b_sample_ipt[
                                                                    ..., atts.index(t_att)] * t_int
            result = self.sess_GAN.run(self.x_sample, feed_dict={self.xa_sample: xa_sample_ipt,
                                                                   self._b_sample: _b_sample_ipt,
                                                                   self.raw_b_sample: raw_a_sample_ipt})
        return np.uint8((result + 1.0) / 2.0 *255)

    def verification(self, path_list, labels, name_list):
        atts = ['Bags_Under_Eyes']
        att_val = [2.0]
        img_size = 256
        n_slide = 10
        test_slide = False
        thres_int = 0.5

        sess_1 = tf.Session()
        te_data = data.MyCeleba(img_size, 1, path_list, labels, part='test', sess=sess_1, crop=False)
        for idx, batch in enumerate(te_data):
            print(idx)
            xa_sample_ipt = batch[0]
            a_sample_ipt = batch[1]
            print(a_sample_ipt)
            b_sample_ipt_list = [a_sample_ipt.copy() for _ in range(n_slide if test_slide else 1)]
            for a in atts:
                i = self.att_default.index(a)
                b_sample_ipt_list[-1][:, i] = 1 - b_sample_ipt_list[-1][:, i]
                b_sample_ipt_list[-1] = data.Celeba.check_attribute_conflict(b_sample_ipt_list[-1],
                                                                             self.att_default[i], self.att_default)
            x_sample_opt_list = [xa_sample_ipt, np.full((1, img_size, img_size // 10, 3), -1.0)]
            raw_a_sample_ipt = a_sample_ipt.copy()
            raw_a_sample_ipt = (raw_a_sample_ipt * 2 - 1) * thres_int
            for i, b_sample_ipt in enumerate(b_sample_ipt_list):
                _b_sample_ipt = (b_sample_ipt * 2 - 1) * thres_int
                if not test_slide:
                    if atts:  # i must be 0
                        for t_att, t_int in zip(atts, att_val):
                            _b_sample_ipt[..., atts.index(t_att)] = _b_sample_ipt[
                                                                        ..., atts.index(t_att)] * t_int
                    if i > 0:  # i == 0 is for reconstruction
                        _b_sample_ipt[..., i - 1] = _b_sample_ipt[..., i - 1] * test_int
                x_sample_opt_list.append(self.sess_GAN.run(self.x_sample, feed_dict={self.xa_sample: xa_sample_ipt,
                                                                       self._b_sample: _b_sample_ipt,
                                                                       self.raw_b_sample: raw_a_sample_ipt}))
            sample = np.concatenate(x_sample_opt_list, 2)

            save_folder = 'sample_testing_multi/' + atts[0]
            save_dir = './output/%s/%s' % (256, save_folder)
            print(save_dir)
            pylib.mkdir(save_dir)
            mp.imsave('%s/%s_' % (save_dir, name_list[idx]) + str(att_val[0]) + '.png', (sample.squeeze(0) + 1.0) / 2.0)

            print('%s/%s_' % (save_dir, name_list[idx]) + str(att_val[0]) + '.png' + ' is done!')

        sess_1.close()


def getPic():
    test_path = './test/crop'
    paths = []
    name_list = os.listdir(test_path)
    pic = np.zeros(shape=(len(name_list), 128, 128, 3), dtype=np.float32)
    for i in range(len(name_list)):
        path = os.path.join(test_path, name_list[i])
        img = mp.imread(path)
        img = cv2.resize(img, (128, 128))
        pic[i, :, :, :] = img / 255.0 * 2.0 - 1.0
        paths.append(path)
    return paths, pic

def check(model):
    _, img = getPic()
    model.sess_cls.as_default()
    model.sess_cls.graph.as_default()
    print(model.sess_cls.run([model.logits], feed_dict={model.input:img}))
    model.sess_GAN.as_default()
    model.sess_GAN.graph.as_default()


if __name__ == '__main__':
    stgan = STGAN()
    # path_list, name_list = stgan.crop(img_size=256)
    img = cv2.imread('./test/crop/0.jpg')
    img = cv2.resize(img, (128, 128))
    labels = stgan.classifier(img)
    # stgan.verification(path_list, labels, name_list)
    result = stgan.inference(img, labels, ["Blond_Hair", "Goatee"])
    print(result)
    cv2.imwrite('./test/result/0.jpg', result[0])
    # plt.show()
    # cv2.waitKey()