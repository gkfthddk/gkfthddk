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

class ptiter(mx.io.DataIter):
    def __init__(self,data_path,data_names,label_names,batch_size=100,begin=0.0,end=1.0,endcut=1,arnum=16,maxx=0.4,maxy=0.4,ptbin=3):
        self.file=rt.TFile(data_path,'read')
        ptread=open('ptjet.txt','r')
        self.ptindx=eval(ptread.readlines()[ptbin])
        ptread.close()
        self.jet=self.file.Get("image")
        self.im = array('b', [0]*(3*(arnum*2+1)*(arnum*2+1)))
        self.jet.SetBranchAddress("image", self.im)
        self.label = array('b', [0])
        self.jet.SetBranchAddress("label", self.label)
        self.Entries=len(self.ptindx)
        self.Begin=int(self.Entries*begin)
        self.End=int(self.Entries*end)
        self.batch_size = batch_size
        self._provide_data = zip(data_names, [(self.batch_size, 3, 33, 33)])
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
        return self.End
    def totalnum(self):
        return int(self.End/self.batch_size)
    def next(self):
        if self.endfile==0:
            arnum=self.arnum
            jetset=[]
            labels=[]
            while True:
                self.jet.GetEntry(self.ptindx[self.ent])
                jetset.append(np.array(self.im).reshape((3,2*arnum+1,2*arnum+1)))
                labels.append(self.label[0])
                self.ent+=1
                if(self.endcut==0 and self.ent>=self.End):
                    self.ent=self.Begin
                    self.endfile=1
                if(len(labels)==self.batch_size):
                    break
            if(self.endcut==1 and int(self.End/self.batch_size)<=int(self.ent/self.batch_size)):
                self.endfile=1

            data=[mx.nd.array(jetset)]
            label=[mx.nd.array(labels)]
            #data = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_data, self.data_gen)]
            #label = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_label, self.label_gen)]
            # print(data)
            return mx.io.DataBatch(data, label)
        else:
            raise StopIteration

