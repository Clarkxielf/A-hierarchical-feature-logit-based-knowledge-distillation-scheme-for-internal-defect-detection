import torch
import torch.nn as nn
from thop import profile
import time
import numpy as np
class VGG19(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG19, self).__init__()

        # 第一层卷积
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 第三层卷积
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 第四层卷积
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # 第五层卷积
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool1d(9)
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 9, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x.unsqueeze(1))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        f=x
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return f, x


model =VGG19()
input1 = torch.randn(1,3500)
flops, params = profile(model, inputs=(input1,))
print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
print('Params = ' + str(params / 1000 ** 2) + 'M')
# model = VGG19()
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
model = VGG19()
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
