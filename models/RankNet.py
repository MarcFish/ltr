import tensorflow as tf
import tensorflow.keras as keras
import argparse
from data import Data
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--file",type=str,default='E:/project/LTR/data/min.txt')
parser.add_argument("--feature_num",type=int,default=46)
parser.add_argument("--lr", type=float,default=1e-3)
parser.add_argument("--batch",type=int,default=1024)
parser.add_argument("--epochs",type=int,default=10)

arg = parser.parse_args()


class RankNet:
    def __init__(self, hidden_units, feature_num, lr=1e-3, session_conf=tf.ConfigProto()):
        self.sess = tf.Session(config = session_conf)
        with tf.variable_scope("model", reuse=None, initializer=tf.initializers.truncated_normal(stddev=0.01)):
            self.input_relevance = tf.placeholder(tf.int32, [None, 1])
            self.input_feature1 = tf.placeholder(tf.float32, [None, feature_num])
            self.input_feature2 = tf.placeholder(tf.float32, [None, feature_num])

            self.net = keras.Sequential()

            for unit in hidden_units:
                self.net.add(keras.layers.Dense(unit))
                self.net.add(keras.layers.LeakyReLU())
                self.net.add(keras.layers.BatchNormalization())

            out1 = self.net(self.input_feature1)
            out2 = self.net(self.input_feature2)
            diff = out1 - out2
            self.loss = -self.input_relevance * diff + tf.math.softplus(diff)
            opt = tf.train.AdagradOptimizer(lr)
            self.train_op = opt.minimize(self.loss)


def get_data(data):
    r_ = list()
    f1_ = list()
    f2_ = list()
    y_ = list()
    for q, qdict in tqdm(data.q_dict.items()):
        rel = qdict['rel']
        feature = qdict['feature']
        for i in range(len(rel)):
            for j in range(len(rel)):
                r = rel[i] - rel[j]
                y = r
                if r < 0:
                    r = 1
                elif r == 0:
                    r = 0.5
                else:
                    r = 0
                r_.append(r)
                f1_.append(feature[i])
                f2_.append(feature[j])
                y_.append(y)
    return r_, f1_, f2_, y_


data = Data(arg.file)
r, f1, f2, y = get_data(data)

# with tf.Graph().as_default():
#     session_conf = tf.ConfigProto()
#     session_conf.gpu_options.allow_growth = True
#     model = RankNet([128, 256, 512], arg.feature_num)
#     model.sess.run(tf.global_variables_initializer())
#     for epoch in range(arg.epochs):
#         pass

