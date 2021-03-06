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


class RankNet(keras.Model):
    def __init__(self, hidden_units):
        super(RankNet, self).__init__()
        self.net = keras.Sequential()
        for unit in hidden_units:
            self.net.add(keras.layers.Dense(unit))
            self.net.add(keras.layers.LeakyReLU())
            self.net.add(keras.layers.BatchNormalization())
        self.net.add(keras.layers.Dense(1, activation='sigmoid'))

    def call(self, inputs):
        f1 = inputs[0]
        f2 = inputs[1]
        diff = self.net(f1) - self.net(f2)
        return diff


def loss(rel, diff):
    return -rel * diff + tf.math.softplus(diff)


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


model = RankNet([128,128,128])
model.compile(loss=loss, optimizer=keras.optimizers.Adam(arg.lr))
model.fit([f1, f2], r, epochs=arg.epochs, batch_size=arg.batch)
