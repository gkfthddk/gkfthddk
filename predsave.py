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

def primes(n):
    primfac = []
    d = 2
    while d*d <= n:
        while (n % d) == 0:
            primfac.append(d)
            n //= d
        d += 1
    if n > 1:
       primfac.append(n)
    return primfac

#python sumpredict.py --network "vgg,*,*,*" --date "2017-09-21,*,*,*" --rat "0.6,0.7,0.8,0.9" --epoch "18,18,17,16" --gpus "1" --save "firsttest"
parser=argparse.ArgumentParser()
parser.add_argument("--begin",type=float,default=0.,help='begin of training must begin<end')
parser.add_argument("--end",type=float,default=1.,help='end of training must begin<end')
parser.add_argument("--rat",type=str,default="1.",help='ratio for qg batch')
parser.add_argument("--date",type=str,default="",help='produced date')
parser.add_argument("--entries",type=int,default=2,help='the number to take batches -1 is get all data')

parser.add_argument("--epoch",type=str,default="0",help='check point number')
parser.add_argument("--load",type=str,default=1,help='loadname')
parser.add_argument("--save",type=str,default=1,help='savename')
parser.add_argument("--ztest",type=int,default=0,help='check point number')
parser.add_argument("--wtest",type=int,default=0,help='check point number')
parser.add_argument("--left",type=str,default="qq100",help='left test')
parser.add_argument("--right",type=str,default="gg100",help='right test')

parser.add_argument("--test",type=str,default="3",help='1 output 2 roc')
fit.add_fit_args(parser)
data.add_data_args(parser)

parser.set_defaults(
    network = "vgg",
    gpus="1",
    num_layers = 18,
    num_classes = 2,
    image_shape = '3,33,33',
    pad_size=4,
    batch_size = 500,
    disp_batched = 99,
    num_epochs = 20,
    lr = .05,
    lr_step_epochs = '10',
)

args=parser.parse_args()
print args

if(args.begin<0. or args.end>1. or args.begin>=args.end):
    print "must be 0 <= begin < end <= 1"
    print args.begin,args.end
    sys.exit(1)
else:
    _beg=args.begin
    _end=args.end
    _mid=(_end-_beg)*5/7+_beg
start=datetime.datetime.now()
qfile=ROOT.TFile("data/ppqq_img.root",'read')
gfile=ROOT.TFile("data/ppgg_img.root",'read')
qjet=qfile.Get("image")
gjet=gfile.Get("image")
ofile=ROOT.TFile("{}.root".format(args.save),'recreate')
qtree=ROOT.TTree("qout","qout")
gtree=ROOT.TTree("gout","gout")
pt=np.zeros(1,dtype=float)
dlout1=np.zeros(1,dtype=float)
qtree.Branch('dlout1',dlout1,'dlout1/D')
gtree.Branch('dlout1',dlout1,'dlout1/D')
dlout2=np.zeros(1,dtype=float)
qtree.Branch('dlout2',dlout2,'dlout2/D')
gtree.Branch('dlout2',dlout2,'dlout2/D')
qtree.Branch('pt',pt,'pt/D')
gtree.Branch('pt',pt,'pt/D')

batch_num=args.batch_size
#train_iter=imiter('../jetimgnum.root',['data'],['softmax_label'],batch_size=batch_num,begin=_beg,end=_mid)

import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
# create a trainable module on GPU 0
# train with the same 
rat=args.rat.split(",")
net=args.network.split(",")
date=args.date.split(",")
epoch=args.epoch.split(",")
test=args.test.split(",")
for i in range(len(rat)):
  if(rat[i]=="*"):
    rat[i]=rat[i-1]
  else:
    rat[i]=eval(rat[i])
  #if(net[i]=="*"):
  #  net[i]=net[i-1]
  #if(date[i]=="*"):
  #  date[i]=date[i-1]
  if(epoch[i]=="*"):
    epoch[i]=epoch[i-1]
  else:
    epoch[i]=eval(epoch[i])


for d in range(len(rat)):
  savename="save/"+args.load+str(rat[d])+"/"+args.save
  if(args.ztest==1):
    test_iter=wkiter(["data/ppzq_img.root","data/ppzg_img.root"],batch_size=args.batch_size,begin=0.,end=1.,istrain=0,friend=0,test=1,endcut=0)
    factor=primes(test_iter.sumnum())
    primebatch=1
    for i in range(len(factor)):
      primebatch=primebatch*factor[i]
      if(primebatch>1010):
        primebatch=primebatch/factor[i]
        break
    args.batch_size=primebatch
    test_iter=wkiter(["data/ppzq_img.root","data/ppzg_img.root"],batch_size=args.batch_size,begin=0.,end=1.,istrain=0,friend=0,test=1,endcut=0)
  else:
    test_iter=wkiter(["data/ppqq_img.root","data/ppgg_img.root"],batch_size=args.batch_size,begin=0.,end=1.,istrain=0,friend=0,test=1,endcut=0)
    primebatch=1
    for i in range(len(factor)):
      primebatch=primebatch*factor[i]
      if(primebatch>1010):
        primebatch=primebatch/factor[i]
        break
    args.batch_size=primebatch
    test_iter=wkiter(["data/ppqq_img.root","data/ppgg_img.root"],batch_size=args.batch_size,begin=0.,end=1.,istrain=0,friend=0,test=1,endcut=0)
    
  if(args.wtest==1):test_iter=wkiter(["data/test"+args.left+"img.root","data/test"+args.right+"img.root"],batch_size=args.batch_size,end=args.end,istrain=1,friend=0)
  if(epoch[d]==0):
    acc=0
    for line in reversed(open("save/"+args.load+str(rat[d])+"/log.log").readlines()):
      if(args.ztest==0):
        indx=line.find("Validation1-accuracy")
        if(indx!=-1):
          accbuf=eval(line[indx+21:])
          if(acc<accbuf):
            acc=accbuf
            epoch[d]=eval(line[line.find("Epoch")+6:line.find("Validation1-ac")-2])+1
      else:
        indx=line.find("Validation2-accuracy")
        if(indx!=-1):
          accbuf=eval(line[indx+21:])
          if(acc<accbuf):
            acc=accbuf
            epoch[d]=eval(line[line.find("Epoch")+6:line.find("Validation2-ac")-2])+1

  sym,arg_params,aux_params=mx.model.load_checkpoint("save/"+args.load+str(rat[d])+"/_",epoch[d])
  #sym,arg_params,aux_params=mx.model.load_checkpoint("save/"+args.load+str(rat[d])+"_"+date[d]+"/_",epoch[d])
    #sym,arg_params,aux_params=mx.model.load_checkpoint("save/jetwcheck_"+net[d]+"_"+date[d]+"/jetcheck",epoch[d])
  
  test_iter.reset()
  inter=sym.get_internals()
  inter=inter[inter.list_outputs()[-3]]
  mod=mx.mod.Module(symbol=inter,context=fit.getctx(args.gpus),label_names=None)
  print test_iter.provide_data, test_iter.provide_label,epoch[d]
  #mod.bind(data_shapes=test_iter.provide_data,label_shapes=test_iter.provide_label)
  mod.bind(data_shapes=test_iter.provide_data)
  mod.init_params()
  mod.set_params(arg_params,aux_params)

  q=[]
  g=[]
  x=[]
  y=[]
  ent=0
  qgauc=0
  xya=0
  qm=0
  gm=0
  entries=args.entries
  start=datetime.datetime.now()
  buftime=datetime.datetime.now()
  if(entries==-1):
      entries=test_iter.totalnum()
  print test_iter.sumnum(), args.batch_size
  print epoch[d],entries,"/",test_iter.totalnum()

  from sklearn.metrics import roc_auc_score, auc,precision_recall_curve,roc_curve,average_precision_score

  for j in range(entries):
    if(j%int(entries/10)==0):print(j)
    a=test_iter.next()
    #print a.label[0].shape
    mod.forward(a)
    b=mod.get_outputs()[0].asnumpy()
    #x=np.append(x,a.label[0].asnumpy())
    #y=np.append(y,b)
    for i in range(len(b)):
      #sys.stdout.write("\r%0.2f"%
      #                (float(100.*ent/entries)))
      #sys.stdout.flush()
      #buftime=datetime.datetime.now()
      #ent+=1
      dlout1[0]=b[i,0]
      dlout2[0]=b[i,1]
      if (a.label[0].asnumpy()[i]==1):
          gjet.GetEntry(gm)
          gm+=1
          pt[0]=gjet.pt
          #g.append(b[i])
          gtree.Fill()
      else:
          qjet.GetEntry(qm)
          qm+=1
          pt[0]=qjet.pt
          #q.append(b[i])
          qtree.Fill()
          
print("saving...")
ofile.Write()
ofile.Close()
print savename+" ","saved"
