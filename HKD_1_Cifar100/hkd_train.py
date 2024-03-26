import torch
import torch.nn as nn

from model import resnet56,resnet110,Sum_weights

import os
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import scipy.io as sio
from losses1 import dkd_loss, FM_LOSS


from dataset import get_cifar100_dataloaders_sample






def train_once_data(num):
    torch.manual_seed(1)
    Sampling_Interval = 1
    split_ratio = 0.8
    BATCH_SIZE = 64
    EPOCH = 240
    LR = 0.05



    train_loader, test_loader, num_data = get_cifar100_dataloaders_sample(64, 64, 0, 4096, mode="exact")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    teacher = resnet110()
    student = resnet56()
    weght = Sum_weights().to(device)

    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        teacher = nn.DataParallel(teacher)#多个GPU训练
        student = nn.DataParallel(student)

    load_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'teacher0.pkl')
    teacher.load_state_dict(torch.load(load_path, map_location=device))
    teacher.to(device)

    student.to(device)


    optimizer = torch.optim.SGD(student.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    loss_task = nn.CrossEntropyLoss()
    loss_fm = FM_LOSS().cuda()


    best_accuracy = 0.
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HKD1_{}.pkl'.format(num))

    writer = SummaryWriter(comment='HKD1')
    Loss = []
    Acc_rate =[]

    for epoch in range(EPOCH):

        if epoch == 150 or epoch == 180 or epoch == 210 :
            LR = LR*0.1
            optimizer = torch.optim.SGD(student.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)


        student.train()
        loss_sum = 0.
        for step, b_y  in enumerate(train_loader):


            f_t, logits_t = teacher(b_y[0].to(device))
            f_s, logits_s = student(b_y[0].to(device))
            [a, b, c] = weght(f_s)
            loss1 = loss_fm(f_s, f_t)
            loss2 = dkd_loss(logits_s, logits_t, b_y[1].to(device), alpha = 1, beta = 2, temperature = 4)
            loss3 = loss_task(logits_s, b_y[1].to(device))
            loss =  a * loss1 + b * loss2 +  c * loss3
            loss_sum += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        Loss.append(loss_sum.data.cpu().numpy())

        student.eval()
        correct = 0
        with torch.no_grad():
            for _, b_y in enumerate(test_loader):

                _, test_output = student(b_y[0].to(device))
                pred_y = torch.max(test_output, dim=1)[1]

                correct += (pred_y == b_y[1].to(device)).sum().item()

            accuracy = correct / len(test_loader.dataset)
            Acc_rate.append(accuracy)
            print('Time: ', num+1, '| Epoch: ', epoch,'|train_loss: %.4f' % loss_sum.item(),
                  '|test_accuracy: %.4f' % accuracy, '({}/{})'.format(correct, len(test_loader.dataset)))

            if EPOCH >= 120 and accuracy>best_accuracy:
                best_accuracy = accuracy
                torch.save(student.state_dict(), save_path)


                for _,  b_y in enumerate(test_loader):

                    _, test_output= student(b_y[0].to(device))
                    pred_y = torch.max(test_output, dim=1)[1]




        writer.add_scalar('train_loss', loss_sum, epoch)
        writer.add_scalar('test_accuracy', accuracy, epoch)

    Loss_data=np.stack(Loss,axis=0)
    Acc_data=np.stack(Acc_rate,axis=0)
    sio.savemat('loss_hkd{}.mat'.format(num),{'loss':Loss_data})
    sio.savemat('accuracy_hkd{}.mat'.format(num),{'accuracy':Acc_data})
    writer.close()


    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


for num in range(0,5):
    train_once_data(num)
