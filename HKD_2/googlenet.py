import torch.nn as nn
import torch
import torch.nn.functional as F
from thop import profile
import time
import numpy as np

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=2, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv1d(1, 64, kernel_size=7, stride=2, padding=3)  # BasicConv2d类
        self.maxpool1 = nn.MaxPool1d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv1d(64, 64, kernel_size=1)
        self.conv3 = BasicConv1d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool1d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)  # Inception类
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool1d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool1d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)  # InceptionAux类
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool1d(9)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 9, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

        if init_weights:
            self._initialize_weights()

    # 正向传播
    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x.unsqueeze(1))  # N x 64 x 112 x 112
        x = self.maxpool1(x)  # N x 64 x 56 x 56

        x = self.conv2(x)  # N x 64 x 56 x 56
        x = self.conv3(x)  # N x 192 x 56 x 56
        x = self.maxpool2(x)  # N x 192 x 28 x 28

        x = self.inception3a(x)  # N x 256 x 28 x 28
        x = self.inception3b(x)  # N x 480 x 28 x 28
        x = self.maxpool3(x)  # N x 480 x 14 x 14

        x = self.inception4a(x)  # N x 512 x 14 x 14
        # if self.training and self.aux_logits:  # eval model不执行该部分
        #     aux1 = self.aux1(x)
        x = self.inception4b(x)  # N x 512 x 14 x 14
        x = self.inception4c(x)  # N x 512 x 14 x 14
        x = self.inception4d(x)  # N x 528 x 14 x 14
        # if self.training and self.aux_logits:  # eval model不执行该部分
        #     aux2 = self.aux2(x)
        x = self.inception4e(x)  # N x 832 x 14 x 14
        x = self.maxpool4(x)  # N x 832 x 7 x 7

        x = self.inception5a(x)  # N x 832 x 7 x 7
        x = self.inception5b(x)  # N x 1024 x 7 x 7

        x = self.avgpool(x)  # N x 1024 x 1 x 1
        f = x
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = torch.flatten(x, 1)  # N x 1024
        # x = self.dropout(x)
        # x = self.fc(x)  # N x 1000 (num_classes)

        # if self.training and self.aux_logits:  # eval model不执行该部分
        #     return x, aux2, aux1
        return f,x

    # 初始化权重
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# 类Inception，有四个分支
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv1d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv1d(in_channels, ch3x3red, kernel_size=1),
            BasicConv1d(ch3x3red, ch3x3, kernel_size=3, padding=1)  # 保证输出大小等于输入大小
        )

        self.branch3 = nn.Sequential(
            BasicConv1d(in_channels, ch5x5red, kernel_size=1),
            BasicConv1d(ch5x5red, ch5x5, kernel_size=5, padding=2)  # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            BasicConv1d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        # 四个分支连接起来
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


# 辅助分类器：类InceptionAux，包括avepool+conv+fc1+fc2
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool1d(kernel_size=5, stride=3)
        self.conv = BasicConv1d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.averagePool(x)  # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)  # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)  # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)  # N x 1024
        x = self.fc2(x)  # N x num_classes
        return x


# 类BasicConv2d，包括conv+relu
class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


# x=torch.randn(1,7000)
# model=resnet10_1d()
# output = model(x)
# # print(output.shape)
# print(model)
# model =GoogLeNet()
# input1 = torch.randn(1,3500)
# flops, params = profile(model, inputs=(input1,))
# print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
# print('Params = ' + str(params / 1000 ** 2) + 'M')
# model = GoogLeNet()
# input = torch.randn(1,3500)
#
# starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# start = time.perf_counter()
# model.to(device)
# input = input.to(device)
# to_gpu = (time.perf_counter() - start) * 1000
#
# # GPU warm-up
# starter.record()
# for _ in range(10):
#     _ = model(input)
# ender.record()
# torch.cuda.synchronize()
# warm_up_time = starter.elapsed_time(ender)
# print("GPU warm up time: ", warm_up_time)
#
# timings = []
# with torch.no_grad():
#     for i in range(100):
#         starter.record()
#         res = model(input)
#         ender.record()
#         # wait for GPU sync
#         torch.cuda.synchronize()
#         curr_timing = starter.elapsed_time(ender)
#         timings.append(round(curr_timing, 3))
# print(timings)
model = GoogLeNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device )
dummy_input = torch.randn(1,3500,dtype=torch.float).to(device)
repetitions = 300
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(10):
    _= model(dummy_input)
# MEASURE PERFORMANCE
with torch.no_grad():
    for rep in range(repetitions):
        # starter.record()
        start=time.time()
        _=model(dummy_input)
        # ender.record()
        end=time.time()
# WAIT FOR GPU SYNC
#         torch.cuda.synchronize()
        curr_time = end-start
        timings[rep]= curr_time
mean_syn =np.sum(timings)/ repetitions
std_syn =np.std(timings)
print(mean_syn)