import torch
import torch.nn as nn
import torch.utils.data as Data
from training_data import load_data
# from teacher import resnet18_1d
from resnet10 import resnet10_1d
from resnet10 import ResNet_1D
import os
from torch.utils.tensorboard import SummaryWriter
from ConfusionMatrix import ConfusionMatrix
from transform import FFT
import numpy as np
import scipy.io as sio
# from losses import dkd_loss, FM_LOSS
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torchvision.models.feature_extraction import create_feature_extractor
font_dict={'family':'Times New Roman','size': 20}

torch.manual_seed(1)
Sampling_Interval = 1
split_ratio = 0.8
BATCH_SIZE = 24
EPOCH = 200
LR = 0.1


# Xtrain_D, Labeltrain_D, Xtest_D, Labeltest_D = load_data('D', split_ratio, Sampling_Interval)
# Xtrain_N, Labeltrain_N, Xtest_N, Labeltest_N = load_data('N', split_ratio, Sampling_Interval)
Xtrain_AD, Labeltrain_AD, Xtest_AD, Labeltest_AD = load_data('AD', split_ratio, Sampling_Interval)
Xtrain_AN, Labeltrain_AN, Xtest_AN, Labeltest_AN = load_data('AN', split_ratio, Sampling_Interval)
Xtrain_BD, Labeltrain_BD, Xtest_BD, Labeltest_BD = load_data('BD', split_ratio, Sampling_Interval)
Xtrain_BN, Labeltrain_BN, Xtest_BN, Labeltest_BN = load_data('BN', split_ratio, Sampling_Interval)
Xtrain_CD, Labeltrain_CD, Xtest_CD, Labeltest_CD = load_data('CD', split_ratio, Sampling_Interval)
Xtrain_CN, Labeltrain_CN, Xtest_CN, Labeltest_CN = load_data('CN', split_ratio, Sampling_Interval)
# Xtrain_DD, Labeltrain_DD, Xtest_DD, Labeltest_DD = load_data('DD', split_ratio, Sampling_Interval)
# Xtrain_DN, Labeltrain_DN, Xtest_DN, Labeltest_DN = load_data('DN', split_ratio, Sampling_Interval)

Xtrain = torch.cat([Xtrain_AD, Xtrain_AN,Xtrain_BD, Xtrain_BN,Xtrain_CD, Xtrain_CN], 0)
Labeltrain = torch.cat([Labeltrain_AD, Labeltrain_AN,Labeltrain_BD, Labeltrain_BN,Labeltrain_CD, Labeltrain_CN], 0)
Xtest = torch.cat([Xtest_AD, Xtest_AN,Xtest_BD, Xtest_BN,Xtest_CD, Xtest_CN], 0)
Labeltest = torch.cat([Labeltest_AD, Labeltest_AN,Labeltest_BD, Labeltest_BN,Labeltest_CD, Labeltest_CN], 0)

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
    batch_size=Xtest.shape[0],
    shuffle=False
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# teacher = resnet18_1d(num_class=2)
student = resnet10_1d(num_class=6)

student.to(device)

if torch.cuda.device_count() >= 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # teacher = nn.DataParallel(teacher)
    student = nn.DataParallel(student)

load_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HKDABC3_0.pkl')
student.load_state_dict(torch.load(load_path, map_location=device), strict=False)  # 加载模型



with torch.no_grad():
    for _, (b_x, b_y) in enumerate(test_loader):
        # _, (b_x, b_y) = enumerate(test_loader)
        x = FFT(b_x)
        f_s, test_output = student(x.to(device))
        f_s = torch.flatten(f_s, 1)
        f_s = f_s.cpu()
        f_s = f_s.numpy()
        label_batch = b_y.to(device).cpu().numpy()

        figure, ax = plt.subplots(figsize=(7,5))

        X_tsne = TSNE(n_components=2, perplexity=45, early_exaggeration=50, learning_rate=1000, n_iter=1000,
                         init='random', random_state=501).fit_transform(f_s)
        X_pca = PCA(n_components=2).fit_transform(f_s)

        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)#归一化
        label_dict = {0: 'AN', 1: 'AD', 2: 'BN', 3: 'BD', 4: 'CN', 5: 'CD'}

        Color = ['blue', 'violet', 'c', 'gold', 'red','green']

        # fig, ax2 = plt.subplots(1, 1, figsize=(8, 5))
        # ax2.scatter(X_norm[:, 0], X_norm[:, 1], c='c', s=30, marker='o')
        for i in range(6):
            if (label_batch == i).any():
                plt.scatter(X_norm[label_batch == i, 0], X_norm[label_batch == i, 1], s=30, c=Color[i], alpha=1, marker='o',label= label_dict[i])

        plt.legend(loc='upper right',prop={'family':'Times New Roman','size':12})
        plt.tick_params(labelsize=20)
        plt.xticks(np.arange(0,1.25,0.25))
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        # plt.axis('off')
        plt.savefig('./transfer_before.png')
        plt.show()
        #
        # plt.figure(figsize=(10, 5))
        # plt.subplot(121)
        # plt.scatter(X_norm[:, 0], X_norm[:, 1], c=label_batch, label="t-SNE")
        # plt.legend()
        # plt.subplot(122)
        # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=label_batch, label="PCA")
        # plt.legend()
        # # plt.savefig('images/digits_tsne-pca.png', dpi=120)
        # plt.show()

