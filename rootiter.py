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

class rootiter(mx.io.DataIter):
    def __init__(self,data_path,data_names,label_names,batch_size=100,arnum=16,maxx=0.4,maxy=0.4):
        self.batch_size = batch_size
        self.ent=0
        self.arnum=arnum
        self.maxx=maxx
        self.maxy=maxy
        self.endfile=0
        self.f=rt.TFile(data_path,'read')
        self.jet=rt.gDirectory.Get('jetAnalyser/jetAnalyser')
        self.Entries=self.jet.GetEntriesFast()
    def __iter__(self):
        return self

    def reset(self):
        self.jet.GetEntry(0)
        self.ent=0
        self.endfile = 0

    def __next__(self):
        return self.next()



    def close(self):
        self.f.Close()
    def printr(self):
        print self.ent,self.jet.pt
    def next(self):
        if self.endfile==0:
            maxx=self.maxx
            maxy=self.maxy
            arnum=self.arnum
            bx=2*maxx/(2*arnum+1)
            by=2*maxy/(2*arnum+1)
            jetset=[]
            labels=[]
            for i in range(self.batch_size):
                if self.ent>=self.Entries:
                    self.endfile=1
                    break
                self.jet.GetEntry(self.ent)
                if self.jet.partonId==0:
                    continue
                palet=np.zeros(shape=(3,2*arnum+1,2*arnum+1))
                r=0
                g=0
                b=0
                ##
                for i in range(len(self.jet.dau_pt)):
                    x=int(math.floor((self.jet.dau_deta[i]/bx)+0.5)+arnum)
                    if x<0 or x>2*arnum:
                        continue
                    y=int(math.floor((self.jet.dau_dphi[i]/by)+0.5)+arnum)
                    if y<0 or y>2*arnum:
                        continue
                    pt=self.jet.dau_pt[i]
                    if self.jet.dau_charge[i]==0:
                        palet[1][x][y]+=pt
                        #overlap[1][x][y]+=pt
                        if palet[1][x][y]>g:
                            g=palet[1][x][y]
                    else:
                        palet[0][x][y]+=pt
                        palet[2][x][y]+=1
                        #overlap[0][x][y]+=pt
                        #overlap[2][x][y]+=1
                        if palet[0][x][y]>r:
                            r=palet[0][x][y]
                        if palet[2][x][y]>b:
                            b=palet[2][x][y]
                        #pass
                #record.write(palet)
                for i in range(2*arnum+1):
                    for j in range(2*arnum+1):
                        if r!=0:
                            palet[0][i][j]=255*palet[0][i][j]/r
                        if g!=0:
                            palet[1][i][j]=255*palet[1][i][j]/g
                        if b!=0:
                            palet[2][i][j]=255*palet[2][i][j]/b
                ##
                jetset.append(palet)
                if self.jet.partonId==21:
                    labels.append(1)
                else:
                    labels.append(0)
                    
                self.ent+=1
            data=[mx.nd.array(jetset)]
            label=[mx.nd.array(labels)]
            #data = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_data, self.data_gen)]
            #label = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_label, self.label_gen)]
            return mx.io.DataBatch(data, label)
        else:
            raise StopIteration
