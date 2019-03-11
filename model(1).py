import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import time


def processed_image(img, scale):
    height, width = img.shape
    new_height = int(height * scale)  # resized new height
    new_width = int(width * scale)  # resized new width
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)  # resized image
    img_resized = (img_resized - 127.5) / 128
    return img_resized

def IoU(box, boxes):
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[2] - boxes[0] + 1) * (boxes[3] - boxes[1] + 1)
    xx1 = np.maximum(box[0], boxes[0])
    yy1 = np.maximum(box[1], boxes[1])
    xx2 = np.minimum(box[2], boxes[2])
    yy2 = np.minimum(box[3], boxes[3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr



class Data():

    def __init__(self):
        if os.path.exists("data_go.npz") == False:
            files = open("表情数据/fer2013.csv", "r", encoding="utf-8")
            datas = [itr.strip().split(',')
                     for itr in files.readlines()[1:]]
            labels = np.array([int(itr[0]) for itr in datas])
            images = np.array([[float(ii) for ii in itr[1].split(' ')]
                               for itr in datas
                               ])
            images = images - np.mean(images, 1, keepdims=True)
            images = images / (np.max(np.abs(images), 1, keepdims=True) + 1e-6)
            images = np.reshape(images, [-1, 48, 48, 1])
            np.savez("data.npz", images=images, labels=labels)
            self.images = images
            self.labels = labels
        else:
            files = np.load("data_go.npz")
            self.images = files['images'][0:100]
            self.labels = files['bbox'][0:100]
            # self.images = files['images']
            # self.labels = files['bbox']
        # print("DataShape:", np.shape(self.images), np.shape(self.labels))
        self.seqlen = len(self.labels)

    def next_batch(self, batch_size=32):
        x = []
        new_idx = []
        idx = np.random.randint(0, self.seqlen, [batch_size])
        for i in idx:
            # print(os.path.exists(self.images[i]))
            if os.path.exists(self.images[i]):
                # print('eixts')
                image = cv2.imread(self.images[i], cv2.IMREAD_GRAYSCALE)
                height, width= image.shape
                if height==512 and width==512:
                    image = processed_image(image, 0.25)
                    image = np.expand_dims(image, axis=2)
                    x.append(image)
                    new_idx.append(i)
            # x = np.array(x)
                d = self.labels[new_idx]*0.25
        return x, d

    def test_data(self):
        return self.images[:10], self.labels[:10]

    


class Model():
    """
    表情识别中使用VGGNet-16作为基本模型
    """

    def __init__(self, batch_size=2, is_training=True):
        """
        初始化类
        """
        self.batch_size = batch_size
        self.is_training = is_training
        self.build_model()
        self.init_sess()

    def build_model(self):
        """
        构建计算图
        """
        self.graph = tf.Graph()

        def block(net, n_conv, n_chl, blockID):
            """
            定义多个CNN组合单元
            """
            with tf.variable_scope("block%d" % blockID):
                for itr in range(n_conv):
                    net = tf.layers.conv2d(net,
                                           n_chl, 3,
                                           activation=tf.nn.relu,
                                           padding="same")
                # net = tf.layers.max_pooling2d(net, 2, 2)
            return net
        with self.graph.as_default():
            # 人脸数据
            self.inputs = tf.placeholder(tf.float32,
                                         [None, 128, 128, 1],
                                         name="inputs")
            # 表情序列，用0-6数字表示
            self.target = tf.placeholder(tf.float32,
                                         [None,4],
                                         name="target")
            net = block(self.inputs, 1, 16, 1)  # 减少网络层数到1层
            net = block(net, 1, 32, 2)
            # net = block(net, 2, 256, 3)
            # net = block(net, 2, 512, 4)
            # net = block(net, 2, 512, 5)
            net = tf.layers.flatten(net)
            net = tf.layers.dense(net, 512, activation=tf.nn.relu)  # 留一个全链接层
            # net = tf.layers.dense(net, 4096, activation=tf.nn.relu)
            self.logits = tf.layers.dense(net, 4, activation=None)
            # 计算精确度

            # print(accuracy.dtype)
            # self.acc = tf.reduce_mean(tf.cast(accuracy, tf.float32))
            # 计算loss
            loss = tf.reduce_sum(tf.square(self.target - self.logits), axis=1)
            self.loss = tf.reduce_mean(loss)
            # 优化
            self.step = tf.train.AdamOptimizer().minimize(self.loss)
            self.all_var = tf.global_variables()
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
    


    def init_sess(self, dirs="model/"):
        """
        初始化会话
        """
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)
        files = tf.train.latest_checkpoint(dirs)
        if files != None:
            self.saver.restore(self.sess, files)

    def train(self):
        datatools = Data()
        ls_list = []
        acc_list = []
        itr_list = []
        for itr in range(100):
            time_data_star = time.time()
            x, d = datatools.next_batch()
            time_data_end = time.time()
            print('readDataTime',time_data_end-time_data_star)
            print('epoch:',itr,'size of batch:',np.shape(d),np.shape(x),d.dtype)
            ls, _ = self.sess.run(
                [self.loss, self.step],
                feed_dict={self.inputs: x,
                           self.target: d
                           }
            )
            print('epoch:',itr,'loss:', ls)
            if itr % 3 == 0:
                pre = self.sess.run(
                    self.logits,
                    feed_dict={self.inputs: x[:10]})
                acc = 0
                for p in range(len(pre)):

                    ovr = IoU(pre[p],d[:10][p])
                    print('epoch:',itr,"IOU:",ovr)
                    if ovr>0.2:
                        acc+=1
                acc=acc/10
                print("Step%d:loss:%f,accuarcy:%f" % (itr, ls, acc))
            #     ls_list.append(ls)
            #     acc_list.append(acc)
            #     itr_list.append(itr)
            if itr%3==0:
                self.saver.save(self.sess, "model/")
        # plt.title('Result Analysis')
        # plt.plot(itr_list, ls_list, color='blue', label='loss')
        # plt.plot(itr_list, acc_list, color='green', label='acc')
        # plt.legend()
        # plt.show()

model = Model()
model.train()
# model.test()
print('tests')