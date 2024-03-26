import torch
import torch.nn as nn
import torch.utils.data as Data
from training_data import load_data
from teacher import resnet18_1d
from resnet10 import resnet10_1d,Sum_weights
import os
from torch.utils.tensorboard import SummaryWriter
from ConfusionMatrix import ConfusionMatrix
from transform import FFT
import numpy as np
import scipy.io as sio
from losses import dkd_loss, FM_LOSS


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

    teacher = resnet18_1d(num_class=2)
    student = resnet10_1d(num_class= 2)

    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        teacher = nn.DataParallel(teacher)
        student = nn.DataParallel(student)

    load_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HKD1_2.pkl')
    teacher.load_state_dict(torch.load(load_path, map_location=device))
    teacher.to(device)
    student.to(device)
    weght = Sum_weights().to(device)


    optimizer = torch.optim.SGD(student.parameters(), lr=LR, momentum=0.9, weight_decay=1e-5)
    loss_task = nn.CrossEntropyLoss()
    loss_fm = FM_LOSS().cuda()

    best_accuracy = 0.
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HKD2_{}.pkl'.format(num))

    writer = SummaryWriter(comment='HKD_2')
    Loss = []
    Acc_rate =[]

    for epoch in range(EPOCH):

        if epoch==(EPOCH-40) or epoch==(EPOCH-20):
            LR = LR*0.1
            optimizer = torch.optim.SGD(student.parameters(), lr=LR, momentum=0.9, weight_decay=1e-5)


        student.train()
        loss_sum = 0.
        for step, (b_x, b_y) in enumerate(train_loader):

            x = FFT(b_x)
            f_t, logits_t = teacher(x.to(device))
            f_s, logits_s = student(x.to(device))
            [a, b, c, d] = weght(f_s)
            loss1 = loss_fm(f_s, f_t)
            loss2 = dkd_loss(logits_s, logits_t, b_y.to(device), alpha = a, beta = b, temperature = 5)
            loss3 = loss_task(logits_s, b_y.to(device))
            loss = c * loss1 + loss2 + d * loss3
            loss_sum += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        Loss.append(loss_sum.data.cpu().numpy())

        student.eval()
        correct = 0
        with torch.no_grad():
            for _, (b_x, b_y) in enumerate(test_loader):
                x = FFT(b_x)
                _, test_output = student(x.to(device))
                pred_y = torch.max(test_output, dim=1)[1]

                correct += (pred_y == b_y.to(device)).sum().item()

            accuracy = correct / len(test_loader.dataset)
            Acc_rate.append(accuracy)
            print('Time: ', num+1, '| Epoch: ', epoch,'|train_loss: %.4f' % loss_sum.item(),
                  '|test_accuracy: %.4f' % accuracy, '({}/{})'.format(correct, len(test_loader.dataset)))

            if accuracy>best_accuracy:
                best_accuracy = accuracy
                torch.save(student.state_dict(), save_path)

                confusion = ConfusionMatrix(num_classes=2, labels=['N', 'D'])
                for _, (b_x, b_y) in enumerate(test_loader):
                    x=FFT(b_x)
                    _, test_output = student(x.to(device))
                    pred_y = torch.max(test_output, dim=1)[1]


                    confusion.update(pred_y.to("cpu").numpy(), b_y.to("cpu").numpy())

        writer.add_scalar('train_loss', loss_sum, epoch)
        writer.add_scalar('test_accuracy', accuracy, epoch)

    Loss_data=np.stack(Loss,axis=0)
    Acc_data=np.stack(Acc_rate,axis=0)
    # sio.savemat('loss_hkd{}.mat'.format(num),{'loss':Loss_data})
    sio.savemat('accuracy_hkd{}.mat'.format(num),{'accuracy':Acc_data})
    writer.close()
    #打开Terminal,键入tensorboard --logdir=logs(logs是事件文件所保存路径)

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    confusion.summary()
    # confusion.plot()
    return best_accuracy

accuracy_rate = []
for num in range(0,5):
    best_acc = train_once_data(num)
    print(best_acc)
    accuracy_rate.append(best_acc)
mean_accuracy = np.mean(accuracy_rate)
std = np.std(accuracy_rate, ddof=1)
print(mean_accuracy, '\n', std)