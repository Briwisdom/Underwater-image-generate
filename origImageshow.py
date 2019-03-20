# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
orig_class='Parrotfish_128'
X_train = np.load('E:/UnderWaterDataset/'+orig_class+'.npy')
X_train = (X_train.astype(np.float32)) / 256

r, c = 5, 5
fig, axs = plt.subplots(r, c)
cnt = 0
for i in range(r):
    for j in range(c):
        axs[i,j].imshow(X_train[cnt, :,:,:])
        axs[i,j].axis('off')
        cnt += 1
fig.savefig('E:/UnderWaterDataset/original/'+orig_class+'.png')
plt.close()
