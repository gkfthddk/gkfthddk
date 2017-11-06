from imiter import *
from wkiter import *
from rootiter import *
import mxnet as mx
import numpy as np
import random
import datetime
import argparse
from common import fit,data
from sklearn.model_selection import train_test_split
from importlib import import_module
#python sumpredict.py --network "vgg,*,*,*" --date "2017-09-21,*,*,*" --rat "0.6,0.7,0.8,0.9" --epoch "18,18,17,16" --gpus "1" --save "firsttest"
parser=argparse.ArgumentParser()
parser.add_argument("--begin",type=float,default=0.,help='begin of training must begin<end')
parser.add_argument("--end",type=float,default=1.,help='end of training must begin<end')
parser.add_argument("--rat",type=str,default=0.7,help='ratio for qg batch')
parser.add_argument("--date",type=str,default="",help='produced date')
parser.add_argument("--batch_size",type=int,default=100,help='the number of each batch')
parser.add_argument("--entries",type=int,default=2,help='the number to take batches -1 is get all data')

parser.add_argument("--epoch",type=str,default=5,help='check point number')
parser.add_argument("--save",type=str,default=1,help='1 likelyhood 2 roc')
parser.add_argument("--test",type=str,default=1,help='1 likelyhood 2 roc')
fit.add_fit_args(parser)
data.add_data_args(parser)

parser.set_defaults(
    network = "vgg",
    gpus=None,
    num_layers = 18,
    num_classes = 2,
    image_shape = '3,33,33',
    pad_size=4,
    batch_size = 100,
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
test_iter=imiter('../jetimgnumcut.root',['data'],['softmax_label'],batch_size=batch_num,begin=_mid,end=_end)
if(args.test=="fd"):
  test_iter=wkiter(['data/mg5_pp_qq_balanced_pt_100_500_','data/mg5_pp_gg_balanced_pt_100_500_'],['data'],['softmax_label'],batch_size=batch_num,begin=_mid,end=_end,friend=20)
if(args.test=="fz"):
  test_iter=wkiter(['data/mg5_pp_zq_passed_pt_100_500_','data/mg5_pp_zg_passed_pt_100_500_'],['data'],['softmax_label'],batch_size=batch_num,begin=_mid,end=_end,friend=50,zboson=1)

import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
# create a trainable module on GPU 0
# train with the same 
savename="save/"+args.save
rat=args.rat.split(",")
net=args.network.split(",")
date=args.date.split(",")
epoch=args.epoch.split(",")
for i in range(len(rat)):
  if(rat[i]=="*"):
    rat[i]=rat[i-1]
  else:
    rat[i]=eval(rat[i])
  if(net[i]=="*"):
    net[i]=net[i-1]
  if(date[i]=="*"):
    date[i]=date[i-1]
  if(epoch[i]=="*"):
    epoch[i]=epoch[i-1]
  else:
    epoch[i]=eval(epoch[i])
for d in range(len(rat)):
  test_iter.reset()
  if(args.test=="f"):
    sym,arg_params,aux_params=mx.model.load_checkpoint("save/jetwcheck_"+net[d]+"_"+date[d]+"/jetcheck",epoch[d])
  elif(rat[d]==None):
    sym,arg_params,aux_params=mx.model.load_checkpoint("save/jeticheck_"+net[d]+"_"+date[d]+"/jetcheck",epoch[d])
  else:
    sym,arg_params,aux_params=mx.model.load_checkpoint("save/jetwcheck_"+str(rat[d])+"_"+net[d]+"_"+date[d]+"/jetcheck",epoch[d])

  mod=mx.mod.Module(symbol=sym,context=fit.getctx(args.gpus))
  print test_iter.provide_data, test_iter.provide_label
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
  print test_iter.totalnum(),entries

  from sklearn.metrics import roc_auc_score, auc,precision_recall_curve,roc_curve,average_precision_score

  if(entries==-1):
      entries=test_iter.totalnum()
  for j in range(entries):
    a=test_iter.next()
    mod.forward(a)
    b=mod.get_outputs()[0].asnumpy()[:,1]
    if(args.save!=2):
      x=np.append(x,a.label[0].asnumpy())
      y=np.append(y,b)
    if(args.save!=1):
      for i in range(batch_num):
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
  plt.savefig(savename+"_"+str(rat[d])+"_"+date[d]+"like.png")
  f=open(savename+"_"+str(rat[d])+"_"+date[d]+"like.dat",'w')
  f.write(str(q)+"\n")
  f.write(str(g))
  f.close()

  like=plt.figure(1)
  plt.hist(q,bins=50,weights=np.ones_like(q)/y2,histtype='step',alpha=0.7,label=str(rat[d])+'quark')
  plt.hist(g,bins=50,weights=np.ones_like(g)/y2,histtype='step',alpha=0.7,label=str(rat[d])+'gluon')
  plt.legend(loc="center left",bbox_to_anchor=(1,0.5))
  #plt.legend(loc=(1.04,0.5))
  #plt.axis((x1,x2,0,1))

  roc=plt.figure(2)
  t_fpr,t_tpr, _ = roc_curve(x,y)
  t_fnr = 1-t_fpr
  train_auc=np.around(auc(t_fpr,t_tpr),4)
  plt.plot(t_tpr,t_fnr,alpha=0.5,label=str(rat[d])+"AUC = {}".format(train_auc),lw=2)
  plt.legend(loc='lower left')
  plt.figure(3+2*d+1)
  plt.plot(t_tpr,t_fnr,alpha=0.5,label=str(rat[d])+"AUC = {}".format(train_auc),lw=2)
  plt.legend(loc='lower left')
  plt.savefig(savename+"_"+str(rat[d])+"_"+date[d]+"_"+str(train_auc)+"roc.png")
  f=open(savename+"_"+str(rat[d])+"_"+date[d]+"roc.dat",'w')
  f.write(str(t_tpr.tolist())+'\n')
  f.write(str(t_fnr.tolist()))
  f.close()
  print datetime.datetime.now()-start
like=plt.figure(1)
plt.savefig(savename+"like",bbox_inches='tight')
roc=plt.figure(2)
plt.savefig(savename+"roc")
argu=open(savename+".txt",'w')
argu.write(str(args))
argu.close()
print savename+" ","saved"
