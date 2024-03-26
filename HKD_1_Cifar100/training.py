
import torch.nn as nn


import torch
import os
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import scipy.io as sio

from dataset import get_cifar100_dataloaders_sample


from model import resnet110



def train_once_data(num):
    torch.manual_seed(1)

    EPOCH = 240
    LR = 0.05



    train_loader, test_loader, num_data=get_cifar100_dataloaders_sample(64, 64, 0, 4096, mode="exact")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model=resnet110()

    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    loss_func = nn.CrossEntropyLoss()

    best_accuracy = 0.
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'teacher{}.pkl'.format(num))

    writer = SummaryWriter(comment='F_resnet50')
    Loss = []
    Acc_rate =[]

    for epoch in range(EPOCH):

        if epoch == 150 or epoch == 180 or epoch == 210:
            LR = LR*0.1
            optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-5)


        model.train()
        loss_sum = 0.
        # list=[]
        for step, b_y in enumerate(train_loader):

            # x=FFT(b_x)
            _,output = model(b_y[0].to(device))
            # vessel=output
            # list.append(vessel)
            loss = loss_func(output, b_y[1].to(device))

            loss_sum += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



        model.eval()
        correct = 0
        with torch.no_grad():
            for _, b_y in enumerate(test_loader):

                _,test_output = model(b_y[0].to(device))
                pred_y = torch.max(test_output, dim=1)[1]

                correct += (pred_y == b_y[1].to(device)).sum().item()

            accuracy = correct / len(test_loader.dataset)
            Acc_rate.append(accuracy)
            print('Time: ', num+1, 'Epoch: ', epoch,'|train_loss: %.4f' % loss_sum.item(),
                  '|test_accuracy: %.4f' % accuracy, '({}/{})'.format(correct, len(test_loader.dataset)))

            if EPOCH >= 120 and accuracy>best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), save_path)



        writer.add_scalar('train_loss', loss_sum, epoch)
        writer.add_scalar('test_accuracy', accuracy, epoch)

    Loss_data=np.stack(Loss,axis=0)
    Acc_data=np.stack(Acc_rate,axis=0)
    sio.savemat('loss{}.mat'.format(num),{'loss':Loss_data})
    sio.savemat('accuracy{}.mat'.format(num),{'accuracy':Acc_data})
    writer.close()
    #打开Terminal,键入tensorboard --logdir=logs(logs是事件文件所保存路径)

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
 

for num in range(0,5):
    train_once_data(num)