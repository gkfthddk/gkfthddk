import mxnet as mx
from sklearn.model_selection import train_test_split
import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import tarfile
import datetime
import warnings
import ROOT as rt
import math
import time
import random
from array import array

start=datetime.datetime.now()

jetset=[]
q=[]
g=[]
read=rt.TFile('../jetimgnum.root')
jet=rt.gDirectory.Get('image')
entries=jet.GetEntriesFast()
for ent in range(entries):
  if(ent%int(entries/999)==0 and ent!=0):
    sys.stdout.write("\r%0.2f%%\t " %
    (float(100.*ent/entries)))
    sys.stdout.flush()
  #if ent>1000:
  #    break
  jet.GetEntry(ent)
  if(jet.pt>100 and abs(jet.eta)<2.4 and jet.nMatchedJets==2):
    if(ord(jet.label)==1):
      g.append([ent,jet.pt])
    else:
      q.append([ent,jet.pt])
"""f1=open("labeljet.txt",'w')
f1.write(str(q[0])+"\n")
f1.write(str(g[0])+"\n")
f1.close()"""
g.sort(key=lambda x:x[1])
q.sort(key=lambda x:x[1])

print len(q)
f2=open("ptsort.txt",'w')
f2.write(str(np.array(q,dtype=int)[:,0].tolist())+"\n")
f2.write(str(np.array(g,dtype=int)[:,0].tolist())+"\n")
f2.close()
print datetime.datetime.now()-start
