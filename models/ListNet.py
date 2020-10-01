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
parser.add_argument("--k", type=int, default=5)

arg = parser.parse_args()


class ListNet(keras.Model):
    def __init__(self, hidden_units):
        super(ListNet, self).__init__()
        self.net = keras.Sequential()
        for unit in hidden_units:
            self.net.add(keras.layers.Dense(unit))
            self.net.add(keras.layers.LeakyReLU())
            self.net.add(keras.layers.BatchNormalization())
        self.net.add(keras.layers.Dense(1, activation='sigmoid'))

    def call(self, inputs):
        o = self.net(inputs)
        return o


def loss(rel, out):
    rel = tf.nn.softmax(rel, axis=-1)
    return tf.math.reduce_sum(rel * tf.nn.log_softmax(out, axis=-1), axis=1)


def get_data(data):
    r_ = list()
    f_ = list()
    for q, qdict in tqdm(data.q_dict.items()):
        rel = np.asarray(qdict['rel'])
        feature = np.asarray(qdict['feature'])
        for i in range(arg.k):
            b = np.random.choice(a=len(rel), size=arg.k)
            r_.append(rel[b])
            f_.append(feature[b])
    return np.asarray(r_).astype(np.float32), np.stack(f_).astype(np.float32)


data = Data(arg.file)
r, f = get_data(data)

model = ListNet([128, 128, 128])
model.compile(loss=loss, optimizer=keras.optimizers.Adam(arg.lr))
model.fit(f, r, epochs=arg.epochs, batch_size=arg.batch)
