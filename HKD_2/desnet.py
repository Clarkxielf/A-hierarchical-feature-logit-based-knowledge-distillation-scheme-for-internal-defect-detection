import torch.nn as nn
import torch
from torchvision import models
from thop import profile
import numpy as np
import time

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate=0):
        super(_DenseLayer, self).__init__()
        self.drop_rate = drop_rate
        self.dense_layer = nn.Sequential(
            nn.BatchNorm1d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=num_input_features, out_channels=bn_size * growth_rate, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=bn_size * growth_rate, out_channels=growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        y = self.dense_layer(x)
        if self.drop_rate > 0:
            y = self.dropout(y)

        return torch.cat([x, y], dim=1)


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate=0):
        super(_DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(_DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class _TransitionLayer(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(_TransitionLayer, self).__init__()
        self.transition_layer = nn.Sequential(
            nn.BatchNorm1d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=num_input_features, out_channels=num_output_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition_layer(x)


class DenseNet(nn.Module):
    def __init__(self, num_init_features=64, growth_rate=32, blocks=(6, 12, 24, 16), bn_size=4, drop_rate=0, num_classes=2):
        super(DenseNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        num_features = num_init_features
        self.layer1 = _DenseBlock(num_layers=blocks[0], num_input_features=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[0] * growth_rate
        self.transtion1 = _TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)

        num_features = num_features // 2
        self.layer2 = _DenseBlock(num_layers=blocks[1], num_input_features=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[1] * growth_rate
        self.transtion2 = _TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)

        num_features = num_features // 2
        self.layer3 = _DenseBlock(num_layers=blocks[2], num_input_features=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[2] * growth_rate
        self.transtion3 = _TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)

        num_features = num_features // 2
        self.layer4 = _DenseBlock(num_layers=blocks[3], num_input_features=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[3] * growth_rate

        self.avgpool = nn.AdaptiveAvgPool1d(9)
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 9, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # nn.Linear(4096*2, 1024),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.features(x.unsqueeze(1))

        x = self.layer1(x)
        x = self.transtion1(x)
        x = self.layer2(x)
        x = self.transtion2(x)
        x = self.layer3(x)
        x = self.transtion3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # x = torch.flatten(x, start_dim=1)
        f = x
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return f,x


def DenseNet121(num_classes):
    return DenseNet(blocks=(6, 12, 24, 16), num_classes=num_classes)

def DenseNet169(num_classes):
    return DenseNet(blocks=(6, 12, 32, 32), num_classes=num_classes)

def DenseNet201(num_classes):
    return DenseNet(blocks=(6, 12, 48, 32), num_classes=num_classes)

def DenseNet264(num_classes):
    return DenseNet(blocks=(6, 12, 64, 48), num_classes=num_classes)

# def read_densenet121():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = models.densenet121(pretrained=True)
#     model.to(device)
#     print(model)


# def get_densenet121(flag, num_classes):
#     if flag:
#         net = models.densenet121(pretrained=True)
#         num_input = net.classifier.in_features
#         net.classifier = nn.Linear(num_input, num_classes)
#     else:
#         net = DenseNet121(num_classes)
#
#     return net


# model = DenseNet121(num_classes=2)
# input1 = torch.randn(1,3500)
# flops, params = profile(model, inputs=(input1,))
# print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
# print('Params = ' + str(params / 1000 ** 2) + 'M')
# model = DenseNet121(num_classes=2)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to(device )
# dummy_input = torch.randn(1,3500,dtype=torch.float).to(device)
# repetitions = 300
# timings=np.zeros((repetitions,1))
# #GPU-WARM-UP
# for _ in range(10):
#     _= model(dummy_input)
# # MEASURE PERFORMANCE
# with torch.no_grad():
#     for rep in range(repetitions):
#         # starter.record()
#         start=time.time()
#         _=model(dummy_input)
#         # ender.record()
#         end=time.time()
# # WAIT FOR GPU SYNC
# #         torch.cuda.synchronize()
#         curr_time = end-start
#         timings[rep]= curr_time
# mean_syn =np.sum(timings)/ repetitions
# std_syn =np.std(timings)
# print(mean_syn)