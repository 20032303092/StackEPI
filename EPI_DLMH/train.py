import math
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
from datetime import datetime

import numpy as np
from keras.callbacks import Callback

from EPI_DLMH.model import get_model


def time_since(start):
    s = time.time() - start
    # s = 62 - start
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)


class roc_callback(Callback):
    def __init__(self, name):
        self.name = name

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights(
            "./model/our_model/%sModel%d.h5" % (self.name, epoch))

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


t1 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

# names = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK','all','all-NHEK']
# name=names[0]
# The data used here is the sequence processed by data_processing.py.
names = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK']
msgs = []
for name in names:
    start = time.time()
    Data_dir = '../data/epivan/%s/' % name
    train = np.load(Data_dir + '%s_train.npz' % name)
    X_en_tra, X_pr_tra, y_tra = train['X_en_tra'], train['X_pr_tra'], train['y_tra']
    model = get_model()
    model.summary()
    print('Traing %s cell line specific model ...' % name)
    back = roc_callback(name=name)
    history = model.fit([X_en_tra, X_pr_tra], y_tra, epochs=90, batch_size=32,
                        callbacks=[back])
    t2 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    print("开始时间:" + t1 + "结束时间：" + t2)
    msg = name + " 开始时间:" + t1 + " 结束时间：" + t2 + " spend time: " + time_since(start)
    print(msg)
    msgs.append(msg)

for msg in msgs:
    print(msg)
