import torch
import scipy.io as sio
import os
import glob
import numpy as np
import torch.utils.data as Data

label_dict = {0:'N', 1:'D'}

# 读取数据
def load_data(LABEL, split_ratio , Sampling_Interval ):
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

    all_Xtrain = []
    all_Labeltrain = []
    all_Xtest = []
    all_Labeltest = []

    for mat_name in glob.glob(os.path.join(DATA_DIR, '*%s.mat' % LABEL)):
        f = sio.loadmat(mat_name)
        data = f['data'][:,:7000].astype('float32')
        if LABEL == 'N':
            label = np.zeros(data.shape[0]).astype('int64')
        elif LABEL=='D':
            label = np.ones(data.shape[0]).astype('int64')

        sample_sequence = [Sampling_Interval * i for i in list(range(0, data.shape[1] // Sampling_Interval))]
        data = data.T[sample_sequence].T

        sample_train = [int((1 / split_ratio) * i) for i in list(range(0, int(split_ratio * data.shape[0])))]

        all_Xtrain.append(data[sample_train])
        all_Labeltrain.append(label[sample_train])
        all_Xtest.append(np.delete(data, sample_train, 0))
        all_Labeltest.append(np.delete(label, sample_train, 0))

    all_Xtrain = torch.from_numpy(np.concatenate(all_Xtrain, axis=0))
    all_Labeltrain = torch.from_numpy(np.concatenate(all_Labeltrain, axis=0))
    all_Xtest = torch.from_numpy(np.concatenate(all_Xtest, axis=0))
    all_Labeltest = torch.from_numpy(np.concatenate(all_Labeltest, axis=0))

    return all_Xtrain, all_Labeltrain, all_Xtest, all_Labeltest

import torch
import torch.nn as nn
import torch.utils.data as Data
from training_data import load_data
from model import resnet50_1d
import os
from torch.utils.tensorboard import SummaryWriter
from ConfusionMatrix import ConfusionMatrix
from transform import FFT



torch.manual_seed(1)
Sampling_Interval = 1
split_ratio = 0.8
BATCH_SIZE = 24
EPOCH = 200
LR = 0.01


Xtrain_D, Labeltrain_D, Xtest_D, Labeltest_D = load_data('D', split_ratio, Sampling_Interval)
Xtrain_N, Labeltrain_N, Xtest_N, Labeltest_N = load_data('N', split_ratio, Sampling_Interval)

Xtrain = torch.cat([Xtrain_D, Xtrain_N], 0)
Labeltrain = torch.cat([Labeltrain_D, Labeltrain_N], 0)
Xtest = torch.cat([Xtest_D, Xtest_N], 0)
Labeltest = torch.cat([Labeltest_D, Labeltest_N], 0)

# 生成数据集
train_data = Data.TensorDataset(
    Xtrain,
    Labeltrain
)
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_data = Data.TensorDataset(
    Xtest,
    Labeltest
)
test_loader = Data.DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)
