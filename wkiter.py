import mxnet as mx
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import tarfile
import datetime
import random
import warnings
import ROOT as rt
import math
from array import array

class wkiter(mx.io.DataIter):
  def __init__(self,data_path,data_names,label_names,batch_size=100,begin=0.0,end=1.0,endcut=1,arnum=16,maxx=0.4,maxy=0.4):
    if(batch_size<100):
      print("batch_size is small it might cause error")
    self.file=rt.TFile(data_path,'read')
    labeljet=open("labeljet.txt",'r')
    self.q=eval(labeljet.readline())
    self.g=eval(labeljet.readline())
    self.jet=self.file.Get("image")
    self.im = array('b', [0]*(3*(arnum*2+1)*(arnum*2+1)))
    self.jet.SetBranchAddress("image", self.im)
    self.label = array('b', [0])
    self.jet.SetBranchAddress("label", self.label)
    self.Entries=self.jet.GetEntriesFast()
    self.Begin=int(begin*len(self.g))
    self.End=int(self.Entries*end)
    self.gBegin=int(begin*len(self.g))
    self.qBegin=int(begin*len(self.q))
    self.gEnd=int(end*len(self.g))
    self.qEnd=int(end*len(self.q))
    self.batch_size = batch_size
    self._provide_data = zip(data_names, [(self.batch_size, 3, 33, 33)])
    self._provide_label = zip(label_names, [(self.batch_size,)])
    self.ent=self.Begin
    self.arnum=arnum
    self.maxx=maxx
    self.maxy=maxy
    self.endfile=0
    self.endcut=endcut
    self.a=self.gBegin
    self.b=self.qBegin
  def __iter__(self):
    return self

  def reset(self):
    self.jet.GetEntry(self.Begin)
    self.ent=self.Begin
    self.a=self.gBegin
    self.b=self.qBegin
    self.endfile = 0

  def __next__(self):
    return self.next()

  @property
  def provide_data(self):
    return self._provide_data

  @property
  def provide_label(self):
    return self._provide_label

  def close(self):
    self.file.Close()
  def printr(self):
    print self.ent,self.jet.pt
  def sampleallnum(self):
    return self.Entries
  def trainnum(self):
    return self.End-self.Begin
  def totalnum(self):
    return int((len(self.q)+len(self.g))/self.batch_size)
  def next(self):
    if self.endfile==0:
      arnum=self.arnum
      jetset=[]
      labels=[]
      rand=random.choice([0.2,0.8])
      for i in range(self.batch_size):
        if(random.random()<rand):
          self.jet.GetEntry(self.g[self.a])
          self.a+=1
          if(self.a>=self.gEnd):
            self.a=self.gBegin
            self.endfile=1
        else:
          self.jet.GetEntry(self.q[self.b])
          self.b+=1
          if(self.b>=self.qEnd):
            self.b=self.gBegin
            self.endfile=1
        jetset.append(np.array(self.im).reshape((3,2*arnum+1,2*arnum+1)))
        labels.append(rand)
        if(self.endcut==0 and self.ent>=self.End):
            self.ent=self.Begin
            self.endfile=1
      data=[mx.nd.array(jetset)]
      label=[mx.nd.array(labels)]
      rand=rand
      #data = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_data, self.data_gen)]
      #label = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_label, self.label_gen)]
      # print(data)
      return mx.io.DataBatch(data, label,rand)
    else:
      raise StopIteration

