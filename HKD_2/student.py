import torch
import torch.nn as nn

class SqueezeExcitation2(nn.Module):
    def __init__(self):
        super(SqueezeExcitation2, self).__init__()
        self.avgpool=nn.AdaptiveAvgPool1d(1)
        self.fc1=nn.Conv1d(256,32,1)
        self.relu=nn.ReLU(inplace=True)
        self.fc2=nn.Conv1d(32,256,1)
        self.sigmod=nn.Sigmoid()


    def forward(self,x):
        scale=self.avgpool(x)
        scale=self.fc1(scale)
        scale=self.relu(scale)
        scale=self.fc2(scale)
        scale=self.sigmod(scale)
        x=scale*x

        return x

class AlexNet_1D(nn.Module):
    def __init__(self,num_class=2,init_weights=True):
        super(AlexNet_1D,self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm1d(192, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm1d(384, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.se = SqueezeExcitation2()
        self.avgpool = nn.AdaptiveAvgPool1d(9)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 9, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, num_class),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x.unsqueeze(1))
        # x =self.se(x)
        x = self.avgpool(x)
        f = x
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return f, x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# x=torch.randn(1,3500)
# mod=AlexNet_1D()
# print(mod(x))