import torch
import os
from training_data import test_loader
from student import resnet18_1d
from transform import FFT
import torch.nn as nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = resnet18_1d(num_class=2).to(device)
if torch.cuda.device_count() >= 1:
  model = nn.DataParallel(model)
load_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HKD1_2.pkl')
model.load_state_dict(torch.load(load_path,map_location=device))

model.eval()
correct = 0
with torch.no_grad():
    for _, (b_x, b_y) in enumerate(test_loader):
        x = FFT(b_x)
        f_t, test_output = model(x.to(device))
        pred_y = torch.max(test_output, dim=1)[1]
        correct += (pred_y == b_y.to(device)).sum().item()

    accuracy = correct / len(test_loader.dataset)
    print('test_accuracy: %.4f' % accuracy, '({}/{})'.format(correct, len(test_loader.dataset)))