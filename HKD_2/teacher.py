import torch
import torch.nn as nn


class SqueezeExcitation2(nn.Module):
    def __init__(self):
        super(SqueezeExcitation2, self).__init__()
        self.avgpool=nn.AdaptiveAvgPool1d(1)
        self.fc1=nn.Conv1d(512,32,1)
        self.relu=nn.ReLU(inplace=True)
        self.fc2=nn.Conv1d(32,512,1)
        self.sigmod=nn.Sigmoid()


    def forward(self,x):
        scale=self.avgpool(x)
        scale=self.fc1(scale)
        scale=self.relu(scale)
        scale=self.fc2(scale)
        scale=self.sigmod(scale)
        x=scale*x

        return x

class BasicBlock_1D(nn.Module):
    expansion=1
    def __init__(self,in_channel,out_channel,stride=1,downsample=None):
        super(BasicBlock_1D, self).__init__()

        self.conv1=nn.Conv1d(in_channels=in_channel,
                             out_channels=out_channel,
                             kernel_size=3,
                             stride=stride,
                             padding=1,
                             bias=False)
        self.bn1=nn.BatchNorm1d(out_channel)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=nn.Conv1d(in_channels=out_channel,
                             out_channels=out_channel,
                             kernel_size=3,
                             stride=1,
                             padding=1,
                             bias=False)
        self.bn2=nn.BatchNorm1d(out_channel)
        self.downsample=downsample

    def forward(self,x):
        identity=x
        if self.downsample is not None:
            identity=self.downsample(x)
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)

        x=self.conv2(x)
        x=self.bn2(x)

        x+=identity
        x=self.relu(x)
        return x

class Bottleneck_1D(nn.Module):
    expansion=4
    def __init__(self,in_channel,out_channel,stride=1,downsample=None):
        super(Bottleneck_1D, self).__init__()

        self.conv1=nn.Conv1d(in_channels=in_channel,
                             out_channels=out_channel,
                             kernel_size=1,
                             stride=1,
                             bias=False)
        self.bn1=nn.BatchNorm1d(out_channel)
        self.conv2=nn.Conv1d(in_channels=out_channel,
                             out_channels=out_channel,
                             kernel_size=3,
                             stride=stride,
                             padding=1,
                             bias=False)
        self.bn2=nn.BatchNorm1d(out_channel)
        self.conv3=nn.Conv1d(in_channels=out_channel,
                             out_channels=out_channel*self.expansion,
                             kernel_size=1,
                             stride=1,
                             bias=False)
        self.bn3=nn.BatchNorm1d(out_channel*self.expansion)
        self.relu=nn.ReLU()
        self.downsample=downsample

    def forward(self,x):
        identity=x
        if self.downsample is not None:
            identity=self.downsample(x)

        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)

        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu(x)

        x=self.conv3(x)
        x=self.bn3(x)

        x+=identity
        x=self.relu(x)
        return x

class ResNet_1D(nn.Module):
    def __init__(self,block,blocks_num,num_class=2,include_top=True):
        super(ResNet_1D,self).__init__()
        self.include_top=include_top
        self.in_channel=64

        self.conv1=nn.Conv1d(in_channels=1,
                             out_channels=self.in_channel,
                             kernel_size=7,
                             stride=2,
                             padding=3,
                             bias=False)
        self.bn1=nn.BatchNorm1d(self.in_channel)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool1d(kernel_size=3,stride=2,padding=1)
        self.layer1=self._make_layer(block,64,blocks_num[0])
        self.layer2=self._make_layer(block,128,blocks_num[1],stride=2)
        self.layer3=self._make_layer(block,256,blocks_num[2],stride=2)
        self.layer4=self._make_layer(block,512,blocks_num[3],stride=2)
        self.se = SqueezeExcitation2()
        if self.include_top:
            self.avgpool=nn.AdaptiveAvgPool1d(9)
            # self.fc=nn.Linear(512*block.expansion,num_class)
            self.classifier = nn.Sequential(
                nn.Linear(512 * 9, 2048),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(2048, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 128),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(128, num_class),
            )
        for m in self.modules():
            if isinstance(m,nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')


    def _make_layer(self,block,channel,block_num,stride=1):
        downsample=None
        if stride!=1 or self.in_channel!=channel*block.expansion:
            downsample=nn.Sequential(
                nn.Conv1d(self.in_channel,
                          channel*block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm1d(channel*block.expansion)
            )
        layers=[]
        layers.append(block(self.in_channel,channel,downsample=downsample,stride=stride))
        self.in_channel=channel*block.expansion

        for _ in range(1,block_num):
            layers.append(block(self.in_channel,channel))

        return nn.Sequential(*layers)
    def forward(self,x):
        x = self.conv1(x.unsqueeze(1))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.se(x)
        if self.include_top:
            x = self.avgpool(x)
            f = x
            x = torch.flatten(x, 1)
            x = self.classifier(x)

        return f, x

def resnet18_1d(num_class=2, include_top=True):
    return ResNet_1D(BasicBlock_1D, [2, 2, 2, 2], num_class=num_class, include_top=include_top)


def resnet34_1d(num_class=2, include_top=True):
    return ResNet_1D(BasicBlock_1D, [3, 4, 6, 3], num_class=num_class, include_top=include_top)


def resnet50_1d(num_class=2, include_top=True):
    return ResNet_1D(Bottleneck_1D, [3, 4, 6, 3], num_class=num_class, include_top=include_top)


def resnet101_1d(num_class=2, include_top=True):
    return ResNet_1D(Bottleneck_1D, [3, 4, 23, 3], num_class=num_class, include_top=include_top)

def resnet152_1d(num_classes=2, include_top=True):
    return ResNet_1D(Bottleneck_1D, [3, 8, 36, 3], num_class=num_classes, include_top=include_top)

# x=torch.randn(1,7000)
# model=resnet18_1d()
# output = model(x)
# print(output)