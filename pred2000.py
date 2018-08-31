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
#python sumpredict.py --network "vgg,*,*,*" --date "2017-09-21,*,*,*" --rat "0.6,0.7,0.8,0.9" --epoch "18,18,17,16" --gpus "1" --save "firsttest"
parser=argparse.ArgumentParser()
parser.add_argument("--begin",type=float,default=0.,help='begin of training must begin<end')
parser.add_argument("--end",type=float,default=1.,help='end of training must begin<end')
parser.add_argument("--rat",type=str,default="0.7",help='ratio for qg batch')
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
  if(args.ztest==1):test_iter=wkiter(["data/ppzq_1000_2000_img.root","data/ppzg_1000_2000_img.root"],batch_size=args.batch_size,begin=0.,end=3./5.+args.end*1./5.,istrain=0,friend=0,test=1)
  else:test_iter=wkiter(["data/ppqq_1000_2000_img.root","data/ppgg_1000_2000_img.root"],batch_size=args.batch_size,begin=3./5.,end=3./5.+args.end*1./5.,istrain=0,friend=0,test=1)
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
  mod=mx.mod.Module(symbol=sym,context=fit.getctx(args.gpus))
  print test_iter.provide_data, test_iter.provide_label,epoch[d]
  mod.bind(data_shapes=test_iter.provide_data,label_shapes=test_iter.provide_label)
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
  print epoch[d],entries,"/",test_iter.totalnum()

  from sklearn.metrics import roc_auc_score, auc,precision_recall_curve,roc_curve,average_precision_score

  if(entries==-1):
      entries=test_iter.totalnum()
  for j in range(entries):
    a=test_iter.next()
    mod.forward(a)
    b=mod.get_outputs()[0].asnumpy()[:,1]
    x=np.append(x,a.label[0].asnumpy())
    y=np.append(y,b)
    for i in range(len(b)):
      #sys.stdout.write("\r%0.2f"%
      #                (float(100.*ent/entries)))
      #sys.stdout.flush()
      #buftime=datetime.datetime.now()
      #ent+=1
      if (a.label[0].asnumpy()[i]==1):
          g.append(b[i])
      else:
          q.append(b[i])
  plt.figure(3+2*d)
  plt.hist(q,bins=50,weights=np.ones_like(q),histtype='step',alpha=0.7,label=str(rat[d])+'quark')
  plt.hist(g,bins=50,weights=np.ones_like(g),histtype='step',alpha=0.7,label=str(rat[d])+'gluon')
  plt.legend(loc="upper center")
  x1,x2,y1,y2=plt.axis()
  plt.savefig(savename+"out.png")
  f=open(savename+"out.dat",'w')
  f.write(str(q)+"\n")
  f.write(str(g))
  f.close()

  out=plt.figure(1)
  plt.hist(q,bins=50,weights=np.ones_like(q)/y2,histtype='step',alpha=0.7,label=str(rat[d])+'quark')
  plt.hist(g,bins=50,weights=np.ones_like(g)/y2,histtype='step',alpha=0.7,label=str(rat[d])+'gluon')
  plt.legend(loc="center left",bbox_to_anchor=(1,0.5))
  #plt.legend(loc=(1.04,0.5))
  #plt.axis((x1,x2,0,1))

  roc=plt.figure(2)
  t_fpr,t_tpr, _ = roc_curve(x,y)
  t_fnr = 1-t_fpr
  train_auc=np.around(auc(t_fpr,t_tpr),4)
  print train_auc
  plt.plot(t_tpr,t_fnr,alpha=0.5,label=str(rat[d])+"AUC = {}".format(train_auc),lw=2)
  plt.legend(loc='lower left')
  plt.figure(3+2*d+1)
  plt.plot(t_tpr,t_fnr,alpha=0.5,label=str(rat[d])+"AUC = {}".format(train_auc),lw=2)
  plt.legend(loc='lower left')
  plt.savefig(savename+str(train_auc)+"roc.png")
  f=open(savename+"roc.dat",'w')
  f.write(str(t_tpr.tolist())+'\n')
  f.write(str(t_fnr.tolist()))
  f.close()
  print datetime.datetime.now()-start
out=plt.figure(1)
plt.savefig(savename+"sumout",bbox_inches='tight')
roc=plt.figure(2)
plt.savefig(savename+"sumroc")
argu=open(savename+".txt",'w')
argu.write(str(args))
argu.close()
print savename+" ","saved"
