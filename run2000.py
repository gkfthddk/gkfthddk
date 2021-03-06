from iter import *
from niter import *
#from wkiter import *
from iter import *
import mxnet as mx
import random
import datetime
import argparse
import sys
import subprocess
from common import fit,data
from sklearn.model_selection import train_test_split
from importlib import import_module
# python run.py --network vgg --end 0.1 --batch-size 100 --num-epochs 10 --gpus "1" --optimizer sgd
#python run.py --end 1 --batch-size 100 --num-epochs 20 --gpus "1" --network vgg --optimizer sgd --rat 0.49
#python run.py --end 1 --batch-size 100 --num-epochs 30 --gpus "0" --network vgg --optimizer sgd --train w --rat 0.8

parser=argparse.ArgumentParser()
parser.add_argument("--memo",default="nothing",help='some memo for memo')
parser.add_argument("--train",default="i",help='i default train w weakly train')
parser.add_argument("--rootdata",default="../jetimgnumcut.root",help='root data')
parser.add_argument("--rand",type=int,default=-1,help='seed of randomfunction')
parser.add_argument("--begin",type=float,default=0.,help='begin of training must begin<end')
parser.add_argument("--end",type=float,default=.1,help='end of training must begin<end')
parser.add_argument("--rat",type=float,default=0.7,help='ratio for weak qg batch')
parser.add_argument("--units",default='3,3,3',help='units')
parser.add_argument("--growth",type=int,default=2,help='growth rate')
parser.add_argument("--save",type=str,default="jetwcheck_",help='savename')
parser.add_argument("--left",type=str,default="qq",help='left train')
parser.add_argument("--right",type=str,default="gg",help='right train')
parser.add_argument("--ztest",type=int,default=0,help='true test zjet')
fit.add_fit_args(parser)
data.add_data_args(parser)

parser.set_defaults(
    network = 'vgg',
    gpus=None,
    num_layers = 13,
    num_classes = 2,
    image_shape = '3,33,33',
    pad_size=4,
    batch_size =100,
    disp_batched =99,
    num_epochs =20,
    optimizer='sgd',
    lr = .05,
    lr_step_epochs='10',
)

args=parser.parse_args()

#print args
#if(args.rand==-1):
#    rand=int(random.random()*10000)
#else:
#    rand=args.rand
#mx.random.seed(rand)

if(args.begin<0. or args.end>1. or args.begin>=args.end):
    print "must be 0 <= begin < end <= 1"
    print args.begin,args.end
    sys.exit(1)
else:
    _beg=args.begin
    _end=args.end
    _mid=(_end-_beg)*5/7+_beg
start=datetime.datetime.now()

rootdata=args.rootdata.split(",")
args.units=[eval(i) for i in args.units.split(",")]


#train_iter=wkiter(["data/train"+args.left+str(int(args.rat*100))+"img.root","data/train"+args.right+str(int(args.rat*100))+"img.root"],batch_size=args.batch_size,end=args.end,istrain=1,friend=0)
train_iter=wkiter(["data/pp{}_1000_2000_img.root".format(args.left),"data/pp{}_1000_2000_img.root".format(args.right)],batch_size=args.batch_size,begin=0./5.,end=0./5.+args.end*3./5.,istrain=1,friend=0)
zjettest_iter=wkiter(["data/ppzq_1000_2000_img.root","data/ppzg_1000_2000_img.root"],batch_size=args.batch_size,begin=4./5.,end=4./5.+args.end*1./5.,istrain=0,friend=0)
dijettest_iter=wkiter(["data/ppqq_1000_2000_img.root","data/ppgg_1000_2000_img.root"],batch_size=args.batch_size,begin=4./5.,end=4./5.+args.end*1./5.,istrain=0,friend=0)
savename="save/"+args.save+str(args.rat)
#savename="save/"+args.save+str(args.rat)+"_"+str(start.date())
#savename="save/"+args.save+str(args.rat)+"_"+args.network+"_"+str(start.date())
net=import_module('symbols.'+args.network)

if(args.network=="dense"):
  sym=net.DenseNet(units=args.units, num_stage=len(args.units), growth_rate=4, num_class=2, reduction=0.5, drop_out=0.2, bottle_neck=True,bn_mom=0.9)
else:
  sym=net.get_symbol(**vars(args))


init=mx.initializer.Xavier('uniform','avg',3)
#init=mx.initializer.MSRAPrelu()
#import logging
#logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
# create a trainable module on GPU 0
if(args.train=="n"):
  model = mx.mod.Module(data_names=('images','variables'),symbol=sym, context=fit.getctx(args.gpus))
else:
  model = mx.mod.Module(symbol=sym, context=fit.getctx(args.gpus))
#model=fit.getmodel(args,sym)
# train with the same 
"""
batch_end_callback = mx.callback.Speedometer(batch_size, 1000),
optimizer_params={'learning_rate':0.1},
"""
#optimizer_params={'learning_rate':0.5,'beta1':0.1,'beta2':0.111},
#batch_end_callback = [mx.callback.Speedometer(100, 1000),mx.callback.ProgressBar],
#optimizer_params={'learning_rate':0.1},
                #batch_end_callback = 
subprocess.call("mkdir "+savename,shell=True)
subprocess.call("rm "+savename+'/log.log',shell=True)
import logging
logging.basicConfig(filename=savename+'/log.log',level=logging.DEBUG)
logging.info(str(args))
logging.info(str(datetime.datetime.now()))
logging.info(str(train_iter.totalnum())+" batches")
argu=open(savename+'/argu.txt','w')
argu.write(str(args))
argu.close()
print train_iter.totalnum(),"batches"
model.fit(train_iter,
                eval_datalist=[dijettest_iter,zjettest_iter],
                initializer=init, 
                optimizer=args.optimizer,
                eval_metric=['accuracy','crossentropy'] if args.network!="dense" else ['acc',mx.metric.create('top_k_accuracy',top_k=5)],
                batch_end_callback =
                [mx.callback.Speedometer(args.batch_size,train_iter.totalnum()/2+1)],
                epoch_end_callback=mx.callback.do_checkpoint(savename+'/_'),
                num_epoch=args.num_epochs)
#lenet_model.save_checkpoint(prefix='jeti1_1',epoch=10)

dijetacc=0
zjetacc=0
for line in reversed(open(savename+"/log.log").readlines()):
  indx=line.find("Validation1-accuracy")
  if(indx!=-1):
    accbuf=eval(line[indx+21:])
    if(dijetacc<accbuf):
      dijetacc=accbuf
      dijetepoch=eval(line[line.find("Epoch")+6:line.find("Validation1-ac")-2])+1
  indx=line.find("Validation2-accuracy")
  if(indx!=-1):
    accbuf=eval(line[indx+21:])
    if(zjetacc<accbuf):
      zjetacc=accbuf
      zjetepoch=eval(line[line.find("Epoch")+6:line.find("Validation2-ac")-2])+1
for i in range(args.num_epochs):
  if(i+1!=dijetepoch and i+1!=zjetepoch):subprocess.call("rm "+savename+'/_-{0:04d}'.format(i+1)+".params",shell=True)

print datetime.datetime.now()-start
