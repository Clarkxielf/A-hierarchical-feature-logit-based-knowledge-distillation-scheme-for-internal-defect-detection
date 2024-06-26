import torch
import torch.nn as nn
import torch.utils.data as Data
from training_data import load_data
from model import resnet50_1d
import os
from torch.utils.tensorboard import SummaryWriter
from ConfusionMatrix import ConfusionMatrix
from transform import FFT
import numpy as np
import scipy.io as sio


def train_once_data(num):
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model =resnet50_1d(num_class=2)

    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-5)
    loss_func = nn.CrossEntropyLoss()

    best_accuracy = 0.
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'teacher{}.pkl'.format(num))

    writer = SummaryWriter(comment='F_resnet50')
    Loss = []
    Acc_rate =[]

    for epoch in range(EPOCH):

        if epoch==(EPOCH-40) or epoch==(EPOCH-20):
            LR = LR*0.1
            optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-5)


        model.train()
        loss_sum = 0.
        for step, (b_x, b_y) in enumerate(train_loader):

            x=FFT(b_x)
            _,output = model(x.to(device))
            loss = loss_func(output, b_y.to(device))
            loss_sum += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        Loss.append(loss_sum.data.cpu().numpy())

        model.eval()
        correct = 0
        with torch.no_grad():
            for _, (b_x, b_y) in enumerate(test_loader):
                x = FFT(b_x)
                _,test_output = model(x.to(device))
                pred_y = torch.max(test_output, dim=1)[1]

                correct += (pred_y == b_y.to(device)).sum().item()

            accuracy = correct / len(test_loader.dataset)
            Acc_rate.append(accuracy)
            print('Time: ', num+1, 'Epoch: ', epoch,'|train_loss: %.4f' % loss_sum.item(),
                  '|test_accuracy: %.4f' % accuracy, '({}/{})'.format(correct, len(test_loader.dataset)))

            if EPOCH >= 120 and accuracy>best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), save_path)

                confusion = ConfusionMatrix(num_classes=2, labels=['N', 'D'])
                for _, (b_x, b_y) in enumerate(test_loader):
                    x=FFT(b_x)
                    _,test_output = model(x.to(device))
                    pred_y = torch.max(test_output, dim=1)[1]


                    confusion.update(pred_y.to("cpu").numpy(), b_y.to("cpu").numpy())

        writer.add_scalar('train_loss', loss_sum, epoch)
        writer.add_scalar('test_accuracy', accuracy, epoch)

    Loss_data=np.stack(Loss,axis=0)
    Acc_data=np.stack(Acc_rate,axis=0)
    sio.savemat('loss{}.mat'.format(num),{'loss':Loss_data})
    sio.savemat('accuracy{}.mat'.format(num),{'accuracy':Acc_data})
    writer.close()
    #打开Terminal,键入tensorboard --logdir=logs(logs是事件文件所保存路径)

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    confusion.summary()
    confusion.plot()

for num in range(0,5):
    train_once_data(num)