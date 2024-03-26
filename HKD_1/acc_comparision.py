import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns

font_dict={'family':'Times New Roman','size': 16}
l1=sio.loadmat('loss_hkd.mat')['loss'][:, 0]
l2=sio.loadmat('loss_hkd.mat')['loss'][:, 1]
l3=sio.loadmat('loss_hkd.mat')['loss'][:, 2]
l4=sio.loadmat('loss_hkd.mat')['loss'][:, 3]
x=np.arange(0,200)
fig,ax=plt.subplots()
plt.plot(x,l1)
plt.plot(x,l2)
plt.plot(x,l3)
plt.plot(x,l4)
plt.xlabel('Epoch',fontdict=font_dict)
plt.ylabel('Loss',fontdict=font_dict)
plt.tick_params(labelsize=14)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
sns.despine()
plt.legend(['loss_hkd','loss_fm','loss_dkd','loss_task'])
# plt.grid()
plt.show()
