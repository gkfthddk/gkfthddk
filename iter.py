import mxnet as mx
import os
import subprocess
import numpy as np
import datetime
import random
import warnings
import ROOT as rt
import math
from array import array

class wkiter(mx.io.DataIter):
  def __init__(self,data_path,data_names=['data'],label_names=['softmax_label'],batch_size=100,begin=0.0,end=1.0,rat=0.7,endcut=1,arnum=16,maxx=0.4,maxy=0.4,istrain=0, fstart=0, friend=0,varbs=0,test=0):
    self.istrain=istrain
    self.rand=0.5
    #if(batch_size<100):
    #  print("batch_size is small it might cause error")
    self.friend=friend
    self.test=test
    print(self.friend,istrain)
    #self.file=rt.TFile(data_path,'read')
    dataname1=data_path[0]
    dataname2=data_path[1]
    self.qfile=rt.TFile(dataname1,'read')
    self.gfile=rt.TFile(dataname2,'read')
    self.qjet=self.qfile.Get("image")
    self.gjet=self.gfile.Get("image")
    self.qim = array('B', [0]*(3*(arnum*2+1)*(arnum*2+1)))
    self.gim = array('B', [0]*(3*(arnum*2+1)*(arnum*2+1)))
    self.qjet.SetBranchAddress("image", self.qim)
    self.gjet.SetBranchAddress("image", self.gim)
    #self.qlabel = array('B', [0])
    #self.glabel = array('B', [0])
    #self.qjet.SetBranchAddress("label", self.qlabel)
    #self.gjet.SetBranchAddress("label", self.glabel)
    self.qEntries=self.qjet.GetEntriesFast()
    self.gEntries=self.gjet.GetEntriesFast()
    self.qBegin=int(begin*self.qEntries)
    self.gBegin=int(begin*self.gEntries)
    self.qEnd=int(self.qEntries*end)
    self.gEnd=int(self.gEntries*end)
    self.a=self.gBegin
    self.b=self.qBegin
    self.ratt=rat
    self.rat=sorted([1-rat,rat])
    self.batch_size = batch_size
    if(varbs==0):
      self._provide_data = zip(data_names, [(self.batch_size, 3, 33, 33)])
    else:
      data_names=['images','variables']
      self._provide_data = zip(data_names, [(self.batch_size, 3, 33, 33),(self.batch_size,5)])
    self.varbs=varbs
    self._provide_label = zip(label_names, [(self.batch_size,)])
    self.arnum=arnum
    self.maxx=maxx
    self.maxy=maxy
    self.endfile=0
    self.endcut=endcut
  def __iter__(self):
    return self

  def reset(self):
    self.rand=0.5
    if(self.friend!=0):
      #print("@@",self.istrain,"g",self.gf,"q",self.qf,"@@")
      for i in range(self.friend):
        self.qjet[i].GetEntry(self.qBegin[i])
        self.gjet[i].GetEntry(self.gBegin[i])
      self.a=self.gBegin[0]
      self.b=self.qBegin[0]
      self.aq=self.gBegin[self.frat]
      self.bg=self.qBegin[self.frat]
      self.gf=0
      self.qf=0
      self.gq=self.frat
      self.qg=self.frat
      self.endfile=0
    else:
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
  def sumnum(self):
    if(self.friend!=0):  
      #a=0
      #for i in range(self.friend):
      #  a+=self.qEnd[i]-self.qBegin[i]+self.gEnd[i]-self.qBegin[i]
      return int((self.qnum+self.gnum))
    else:
      return int((self.qEnd-self.qBegin+self.gEnd-self.gBegin))
  def totalnum(self):
    if(self.friend!=0):  
      #a=0
      #for i in range(self.friend):
      #  a+=self.qEnd[i]-self.qBegin[i]+self.gEnd[i]-self.qBegin[i]
      return int((self.qnum+self.gnum)/self.batch_size*1.)
    else:
      return int((self.qEnd-self.qBegin+self.gEnd-self.gBegin)/self.batch_size*1.)
  def next(self):
    while self.endfile==0:
      arnum=self.arnum
      jetset=[]
      variables=[]
      labels=[]
      rand=random.choice(self.rat)
      if(self.friend!=0):
        rand=self.gnum/1./(self.qnum+self.gnum)
      for i in range(self.batch_size):
        if(self.ratt<1.):
          li=0
        if(self.ratt==1.):
          li=0
        if(random.random()<self.rand):
          self.gjet.GetEntry(self.a)
          self.a+=1
          jetset.append(np.array(self.gim).reshape((3,2*arnum+1,2*arnum+1)))
          labels.append(1-li)
          if(self.a>=self.gEnd):
            if(self.test==1):
              if(self.rand==1):
                self.endfile=1
                break
              self.rand=0
            else:
              self.a=self.gBegin
              self.endfile=1
        else:
          self.qjet.GetEntry(self.b)
          self.b+=1
          jetset.append(np.array(self.qim).reshape((3,2*arnum+1,2*arnum+1)))
          labels.append(0+li)
          if(self.b>=self.qEnd):
            if(self.test==1):
              if(self.rand==0):
                self.endfile=1
                break
              self.rand=1
            else:
              self.b=self.qBegin
              self.endfile=1
        #if(rand<0.5):
        #    labels.append(0)
        #else:
        #    labels.append(1)
        #if(self.endcut==0 and self.ent>=self.End):
            #self.ent=self.Begin
            #self.endfile=1
      if(self.varbs==1):
        data=[mx.nd.array(jetset),mx.nd.array(variables)]
      else:
        data=[mx.nd.array(jetset)]
      label=[mx.nd.array(labels)]
      return mx.io.DataBatch(data, label)
    else:
      if(self.istrain==1):
        print "\n",datetime.datetime.now()  
      raise StopIteration

