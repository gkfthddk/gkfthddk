import os
import subprocess
import numpy as np
import datetime
import random
import warnings
import ROOT as rt
import math
from array import array
import matplotlib.pyplot as plt
dataname1="data/ppqq_img.root"
dataname2="data/ppgg_img.root"
qfile=rt.TFile(dataname1,'read')
gfile=rt.TFile(dataname2,'read')
qjet=qfile.Get("image")
gjet=gfile.Get("image")
qim = array('B', [0]*(3*(33)*(33)))
gim = array('B', [0]*(3*(33)*(33)))
qjet.SetBranchAddress("image", qim)
gjet.SetBranchAddress("image", gim)
qEntries=qjet.GetEntriesFast()
gEntries=gjet.GetEntriesFast()

def draw(num):
  qjet.GetEntry(num)
  im=np.swapaxes(np.swapaxes(np.array(qim).reshape(3,33,33),0,1),1,2)
  plt.imshow(im)

