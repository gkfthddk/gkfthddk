from imiter import *
import numpy as np
import matplotlib.pyplot as plt
train11=imiter('data/pp_gg_default_1_img.root')
train12=imiter('data/pp_gg_default_2_img.root')
train21=imiter('data/pp_qq_default_1_img.root')
train22=imiter('data/pp_qq_default_2_img.root')

g=train11.sumimage(8000)+train12.sumimage(8000)
q=train21.sumimage(8000)+train22.sumimage(8000)
g=g.reshape((3,33,33))/max(g)
q=q.reshape((3,33,33))/max(q)
fig1=plt.figure()
plt.imshow(np.swapaxes(g,0,2),interpolation='none')
fig2=plt.figure()
plt.imshow(np.swapaxes(q,0,2),interpolation='none')

plt.show()
