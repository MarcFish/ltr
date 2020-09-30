import tensorflow as tf
import tensorflow.keras as keras
import argparse
from data import Data
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--file",type=str,default='E:/project/LTR/data/min.txt')
parser.add_argument("--feature_num",type=int,default=46)
parser.add_argument("--lr", type=float,default=1e-3)
parser.add_argument("--batch",type=int,default=1024)
parser.add_argument("--epochs",type=int,default=10)

arg = parser.parse_args()


class RankNet:
    def __init__(self, hidden_units, feature_num, lr=1e-3):
        self.sess = tf.Session()
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
    return np.asarray(r_), np.stack(f1_), np.stack(f2_)


data = Data(arg.file)
r, f1, f2 = get_data(data)


with tf.Graph().as_default():
    model = RankNet([128, 256, 512], arg.feature_num)
    model.sess.run(tf.global_variables_initializer())
    for epoch in range(arg.epochs):

        shuffle_indices = np.random.permutation(np.arange(len(r)))
        r = r[shuffle_indices]
        f1 = f1[shuffle_indices]
        f2 = f2[shuffle_indices]
        y = y[shuffle_indices]
        ll = int(len(r) / arg.batch)
        losss = list()
        for batch_num in range(ll):
            start_index = batch_num * arg.batch
            end_index = min((batch_num + 1) * arg.batch, len(r))

            r_batch = r[start_index:end_index]
            f1_batch = f1[start_index:end_index]
            f2_batch = f2[start_index:end_index]
            y_batch = y[start_index:end_index]

            feed_dict = {
                model.input_relevance:r_batch,
                model.input_feature1:f1_batch,
                model.input_feature2:f2_batch
            }
            _, loss = model.sess.run([model.loss, model.train_op], feed_dict)
            losss.append(loss)
        loss = sum(loss)
    print("epoch:{} loss:{}".format(epoch,loss))
