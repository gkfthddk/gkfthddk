from imiter import *
from wkiter import *
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
parser=argparse.ArgumentParser()
parser.add_argument("--train",default="i",help='i default train w weakly train')
parser.add_argument("--rootdata",default="../jetimgnumcut.root",help='root data')
parser.add_argument("--rand",type=int,default=-1,help='seed of randomfunction')
parser.add_argument("--begin",type=float,default=0.,help='begin of training must begin<end')
parser.add_argument("--end",type=float,default=.1,help='end of training must begin<end')
parser.add_argument("--rat",type=float,default=0.7,help='ratio for weak qg batch')
fit.add_fit_args(parser)
data.add_data_args(parser)

parser.set_defaults(
    network = 'vgg',
    gpus=None,
    num_layers = 18,
    num_classes = 2,
    image_shape = '3,33,33',
    pad_size=4,
    batch_size =100,
    disp_batched =99,
    num_epochs =20,
    optimizer='adagrad',
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

rootdata=args.rootdata

# train_iter =rootiter('/home/gkfthddk/tutorials/gkfthddk/../jet1.root',['data'],['softmax_label'],batch_size=1000,begin=0,end=0.01)
# test_iter =rootiter('/home/gkfthddk/tutorials/gkfthddk/../jet1.root',['data'],['softmax_label'],batch_size=1000,begin=0.01,end=0.012)
batch_num=args.batch_size
if(args.train=="w"):
  train_iter=wkiter(rootdata,batch_size=batch_num,begin=_beg,end=_mid,rat=args.rat,istrain=1)
  savename="save/jetwcheck_"+str(args.rat)+"_"+args.network+"_"+str(start.date())
if(args.train=="i"):
  train_iter=imiter(rootdata,batch_size=batch_num,begin=_beg,end=_mid,istrain=1)
  savename="save/jeticheck_"+args.network+"_"+str(start.date())


test_iter=imiter('../jetimgnumcut.root',batch_size=batch_num,begin=_mid,end=_end)

net=import_module('symbols.'+args.network)

sym=net.get_symbol(**vars(args))

init=mx.initializer.Xavier('uniform','avg',3)
#init=mx.initializer.MSRAPrelu()

#import logging
#logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
# create a trainable module on GPU 0
if("vgg"==args.network):
  print("true")
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
import logging
logging.basicConfig(filename=savename+'/log.log',level=logging.DEBUG)
logging.info(str(args))
argu=open(savename+'/argu.txt','w')
argu.write(str(args))
argu.close()
print train_iter.totalnum(),"batches"
model.fit(train_iter,
                eval_data=test_iter,
                initializer=init, 
                optimizer=args.optimizer,
                eval_metric='acc',
                batch_end_callback =
                [mx.callback.ProgressBar(train_iter.totalnum()),mx.callback.Speedometer(batch_num,train_iter.totalnum()-1)],
                epoch_end_callback=mx.callback.do_checkpoint(savename+'/jetcheck'),
                num_epoch=args.num_epochs)
#lenet_model.save_checkpoint(prefix='jeti1_1',epoch=10)

print datetime.datetime.now()-start
