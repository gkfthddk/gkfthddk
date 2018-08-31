from iter import *
import mxnet as mx
import numpy as np
import random
import datetime
import argparse
import matplotlib.pyplot as plt
from common import fit,data
from sklearn.model_selection import train_test_split
from importlib import import_module
import ROOT
#python sumpredict.py --network "vgg,*,*,*" --date "2017-09-21,*,*,*" --rat "0.6,0.7,0.8,0.9" --epoch "18,18,17,16" --gpus "1" --save "firsttest"
start=datetime.datetime.now()
qfile=ROOT.TFile("data/ppzq_img.root",'read')
gfile=ROOT.TFile("data/ppzg_img.root",'read')
ifile=ROOT.TFile("zqzg.root",'read')
qjet=qfile.Get("image")
gjet=gfile.Get("image")
qout=ifile.Get("qout")
gout=ifile.Get("gout")

for i in range(qjet.GetEntries()):

  qjet.GetEntry(i)
  qout.GetEntry(i)
  if(qjet.pt!=qout.pt):
    
    print("entry {}  in {} out {}".format(i,qjet.pt,qout.pt))
    
    break

