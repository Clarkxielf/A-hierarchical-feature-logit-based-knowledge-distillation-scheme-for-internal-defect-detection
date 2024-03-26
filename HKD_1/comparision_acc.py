import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns

font_dict={'family':'Times New Roman','size': 16}
l1=sio.loadmat('accuracy_teacher.mat')['accuracy'][0]
l2=sio.loadmat('accuracy_hkd2.mat')['accuracy'][0]
l3=sio.loadmat('accuracy_student.mat')['accuracy'][0]
# l4=sio.loadmat('loss4.mat')['loss'][0]
x=np.arange(0,200)
fig,ax=plt.subplots()
plt.plot(x,l1*100)
plt.plot(x,l2*100)
plt.plot(x,l3*100)
# plt.plot(x,l4)
plt.xlabel('Epoch',fontdict=font_dict)
plt.ylabel('Accuracy(%)',fontdict=font_dict)
plt.tick_params(labelsize=14)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
sns.despine()
plt.legend(['Teacher','HFLKD1','Student'])
# plt.grid()
plt.show()
