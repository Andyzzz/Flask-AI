#encoding=utf-8

import tensorflow as tf
from att_classification.models import classifier
import cv2
import os
import numpy as np

test_path = '../test'
model_path = './checkpoints'

att_dict = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2,
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

att_default = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses',
               'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']

def getPic():
    name_list = os.listdir(test_path)
    pic = np.zeros(shape=(len(name_list), 128, 128, 3), dtype=np.float32)
    for i in range(len(name_list)):
        path = os.path.join(test_path, name_list[i])
        img = cv2.imread(path)
        img = cv2.resize(img, (128, 128))
        pic[i, :, :, :] = img / 255.0 * 2.0 - 1.0
    return name_list, pic


def predict():
    input = tf.placeholder(shape=(1, 128, 128, 3), dtype=tf.float32)
    logits = classifier(input, reuse=False, training=False)
    pred = tf.cast(tf.round(tf.nn.sigmoid(logits)), tf.int64)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, os.path.join(model_path, '128.ckpt'))

        name_list, pic = getPic()

        for i in range(len(name_list)):
            print(name_list[i] + ' : ')
            img = pic[i, :, :, :]
            img = np.reshape(img, newshape=(1, 128, 128, 3))
            pre = sess.run(pred, feed_dict={input:img})
            print(pre)


if __name__ == '__main__':
    predict()