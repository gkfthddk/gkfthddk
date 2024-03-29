import mxnet as mx
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import tarfile
import datetime
import warnings
import ROOT as rt
import math
from array import array

class niter(mx.io.DataIter):
    def __init__(self,data_path,data_names=['images','variables'],label_names=['softmax_label'],batch_size=100,begin=0.0,end=1.0,endcut=1,arnum=16,maxx=0.4,maxy=0.4,istrain=0):
        self.istrain=istrain
        if(batch_size<100):
            print("batch_size is small it might cause error")
        self.file=rt.TFile(data_path,'read')
        self.jet=self.file.Get("image")
        self.im = array('B', [0]*(3*(arnum*2+1)*(arnum*2+1)))
        self.jet.SetBranchAddress("image", self.im)
        self.label = array('B', [0])
        self.jet.SetBranchAddress("label", self.label)
        self.Entries=self.jet.GetEntriesFast()
        self.Begin=int(self.Entries*begin)
        self.End=int(self.Entries*end)
        self.batch_size = batch_size
        self._provide_data = zip(data_names, [(self.batch_size, 3, 33, 33),(self.batch_size,3)])
        self._provide_label = zip(label_names, [(self.batch_size,)])
        self.ent=self.Begin
        self.arnum=arnum
        self.maxx=maxx
        self.maxy=maxy
        self.endfile=0
        self.endcut=endcut
    def __iter__(self):
        return self

    def reset(self):
        self.jet.GetEntry(self.Begin)
        self.ent=self.Begin
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
        return int((self.End-self.Begin)/self.batch_size)
    def next(self):
        if self.endfile==0:
            arnum=self.arnum
            jetset=[]
            ptset=[]
            variables=[]
            labels=[]
            """for i in range(self.batch_size):
                self.jet.GetEntry(self.ent)
                jetset.append(np.array(self.im).reshape((3,2*arnum+1,2*arnum+1)))
                labels.append(self.label[0])
                self.ent+=1
                if(self.endcut==0 and self.ent>=self.End):
                    self.ent=self.Begin
                    self.endfile=1"""
            while True:
                self.jet.GetEntry(self.ent)
                self.ent+=1
                if(self.jet.pt<100 or abs(self.jet.eta)>2.4 or self.jet.nMatchedJets!=2):
                    pass
                else:
                    jetset.append(np.array(self.im).reshape((3,2*arnum+1,2*arnum+1)))
                    ptset.append([100*self.label[0]+100,self.label[0],self.label[0]])
                    variables.append([self.jet.ptD,self.jet.axis1,self.jet.axis2,self.jet.nmult,self.jet.cmult])
                    labels.append(self.label[0])
                if(len(labels)>=self.batch_size):
                    break
                #if(self.endcut==0 and self.ent>=self.End):
                if(self.ent>=self.End):
                    self.ent=self.Begin
                    self.endfile=1
            if(self.endcut==1 and int((self.End-self.Begin)/self.batch_size)<=int((self.ent-self.Begin)/self.batch_size)):
                self.endfile=1
            data=[mx.nd.array(jetset),mx.nd.array(variables)]
            label=[mx.nd.array(labels)]
            #data = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_data, self.data_gen)]
            #label = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_label, self.label_gen)]
            # print(data)
            return mx.io.DataBatch(data, label,)
        else:
            raise StopIteration

