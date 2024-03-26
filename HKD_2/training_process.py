import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns


font_dict={'family':'Times New Roman','size': 16}
y1=sio.loadmat('loss_student.mat')['loss'][0]
y2=sio.loadmat('accuracy_student.mat')['accuracy'][0]
x=np.arange(0,200)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1,'r',label="training_loss")
# plt.legend(['training_loss'])
# ax1.legend(loc=4)
# ax1.set_ylabel('Y values for exp(-x)')
ax2 = ax1.twinx() # this is the important function
ax2.plot(x, y2, 'g',label = "test_accuracy")
# plt.legend(['test_accuracy'])
fig.legend(loc=1, bbox_to_anchor=(1,0.5), bbox_transform=ax1.transAxes)
# ax2.set_xlim([0, np.e]);
# ax2.set_ylabel('Y values for ln(x)')
# ax2.set_xlabel('Same X for both exp(-x) and ln(x)')
plt.show()