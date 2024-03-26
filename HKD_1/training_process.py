import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns


font_dict={'family':'Times New Roman','size': 16}
y1=sio.loadmat('loss2.mat')['loss'][0]
y2=sio.loadmat('accuracy_teacher.mat')['accuracy'][0]
x=np.arange(0,200)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1,'r',label="training_loss")
plt.xlabel('Epoch', fontdict=font_dict)
ax1.set_ylabel('Loss', fontdict=font_dict)
plt.tick_params(labelsize=14)
# plt.legend(['training_loss'])
# ax1.legend(loc=4)
# ax1.set_ylabel('Y values for exp(-x)')
ax2 = ax1.twinx() # this is the important function
ax2.plot(x, y2*100, 'g',label = "test_accuracy")
ax2.set_ylabel('Accuracy(%)', fontdict=font_dict)
plt.tick_params(labelsize=14)

# plt.legend(['test_accuracy'])
fig.legend(loc=1, bbox_to_anchor=(1,0.5), bbox_transform=ax1.transAxes)
labels1 = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels1]
labels2 = ax2.get_xticklabels() + ax2.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels2]
# ax2.set_xlim([0, np.e]);
# ax2.set_ylabel('Y values for ln(x)')
# ax2.set_xlabel('Same X for both exp(-x) and ln(x)')
plt.show()