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
  def __init__(self,data_path,data_names=['data'],label_names=['softmax_label'],batch_size=100,begin=0.0,end=1.0,rat=0.7,endcut=1,arnum=16,maxx=0.4,maxy=0.4,istrain=0):
    self.istrain=istrain
    if(batch_size<100):
      print("batch_size is small it might cause error")
    self.file=rt.TFile(data_path,'read')
    self.qjet=self.file.Get("qimage")
    self.gjet=self.file.Get("gimage")
    self.rat=sorted([1-rat,rat])
    self.qim = array('B', [0]*(3*(arnum*2+1)*(arnum*2+1)))
    self.gim = array('B', [0]*(3*(arnum*2+1)*(arnum*2+1)))
    self.qjet.SetBranchAddress("image", self.qim)
    self.gjet.SetBranchAddress("image", self.gim)
    self.qlabel = array('B', [0])
    self.glabel = array('B', [0])
    self.qjet.SetBranchAddress("label", self.qlabel)
    self.gjet.SetBranchAddress("label", self.glabel)
    self.qEntries=self.qjet.GetEntriesFast()
    self.gEntries=self.gjet.GetEntriesFast()
    self.qBegin=int(begin*self.qEntries)
    self.gBegin=int(begin*self.gEntries)
    self.qEnd=int(self.qEntries*end)
    self.gEnd=int(self.gEntries*end)
    self.batch_size = batch_size
    self._provide_data = zip(data_names, [(self.batch_size, 3, 33, 33)])
    self._provide_label = zip(label_names, [(self.batch_size,)])
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
    self.qjet.GetEntry(self.qBegin)
    self.gjet.GetEntry(self.gBegin)
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
  def sampleallnum(self):
    return self.Entries
  def trainnum(self):
    return self.End-self.Begin
  def totalnum(self):
    return int((self.qEnd-self.qBegin+self.gEnd-self.qBegin)/self.batch_size)
  def next(self):
    if self.endfile==0:
      arnum=self.arnum
      jetset=[]
      labels=[]
      rand=random.choice(self.rat)
      for i in range(self.batch_size):
        if(random.random()<rand):
          self.gjet.GetEntry(self.a)
          self.a+=1
          jetset.append(np.array(self.gim).reshape((3,2*arnum+1,2*arnum+1)))
          if(self.a>=self.gEnd):
            self.a=self.gBegin
            self.endfile=1
        else:
          self.qjet.GetEntry(self.b)
          self.b+=1
          jetset.append(np.array(self.qim).reshape((3,2*arnum+1,2*arnum+1)))
          if(self.b>=self.qEnd):
            self.b=self.gBegin
            self.endfile=1
        if(rand<0.5):
            labels.append(0)
        else:
            labels.append(1)
        #if(self.endcut==0 and self.ent>=self.End):
            #self.ent=self.Begin
            #self.endfile=1
      data=[mx.nd.array(jetset)]
      label=[mx.nd.array(labels)]
      rand=rand
      #data = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_data, self.data_gen)]
      #label = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_label, self.label_gen)]
      # print(data)
      return mx.io.DataBatch(data, label)
    else:
      #if(self.istrain==1):
      #  print "\n",datetime.datetime.now()  
      raise StopIteration

