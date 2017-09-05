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
labels=[]
pt=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
maxx=0.4
maxy=0.4                        
arnum=16
bx=2*maxx/(2*arnum+1)
by=2*maxy/(2*arnum+1)
read=rt.TFile('../jetall.root')
jet=rt.gDirectory.Get('jetAnalyser/jetAnalyser')
entries=jet.GetEntriesFast()
for ent in range(entries):
    if(ent%int(entries/999)==0 and ent!=0):
        sys.stdout.write("\r%0.2f%%\t " %
        (float(100.*ent/entries)))
	sys.stdout.flush()
    #if ent>1000:
    #    break
    jet.GetEntry(ent)
    jetpt=jet.pt
    if(jetpt<2000):     
        pt[int(jetpt/100)].append(ent)
    else:
        pt[20].append(ent)

print len(pt)
f=open("ptjet.txt",'w')
for i in range(len(pt)):
    f.write(str(pt[i])+"\n")
f.close()
print datetime.datetime.now()-start
