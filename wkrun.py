from imiter import *
from wkiter import *
import mxnet as mx
import random
import datetime
import argparse
import sys
from common import fit,data
from sklearn.model_selection import train_test_split
from importlib import import_module
#from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, roc_curve, average_precision_score
parser=argparse.ArgumentParser()
#parser.add_argument("--batch_size",type=int,default=100,help='the number of batch size')
#parser.add_argument("--num_epochs",type=int,default=10,help='the number of total epochs')
#parser.add_argument("--gpus",default=None,help='the ports of gpus')
#parser.add_argument("--cpus",default=None,help='the ports of cpus')
parser.add_argument("--begin",type=float,default=0.,help='begin of training must begin<end')
parser.add_argument("--end",type=float,default=.1,help='end of training must begin<end')
#parser.add_argument("--optimizer",default="adagrad",help='the optimizer at fitting')
#parser.add_argument("--network",default="vgg",help='the network at fitting')
fit.add_fit_args(parser)
data.add_data_args(parser)


parser.set_defaults(
    network = 'wkvgg',
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

# train_iter =rootiter('/home/gkfthddk/tutorials/gkfthddk/../jet1.root',['data'],['softmax_label'],batch_size=1000,begin=0,end=0.01)
# test_iter =rootiter('/home/gkfthddk/tutorials/gkfthddk/../jet1.root',['data'],['softmax_label'],batch_size=1000,begin=0.01,end=0.012)
batch_num=args.batch_size
print batch_num
train_iter=wkiter('../jetimgnum.root',['data'],['softmax_label'],batch_size=batch_num,begin=_beg,end=_mid)
test_iter=imiter('../jetimgnum.root',['data'],['softmax_label'],batch_size=batch_num,begin=_mid,end=_end)

net=import_module('symbols.'+args.network)
sym=net.get_symbol(**vars(args))

print "vgg"


import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
# create a trainable module on GPU 0
print args.network
if("vgg"==args.network):
  print("true")
model = mx.mod.Module(symbol=sym, context=fit.getctx(args.gpus))
#model=fit.getmodel(args,sym)
print "gpu pass"
# train with the same 
"""
batch_end_callback = mx.callback.Speedometer(batch_size, 1000),
optimizer_params={'learning_rate':0.1},
"""
#optimizer_params={'learning_rate':0.5,'beta1':0.1,'beta2':0.111},
#batch_end_callback = [mx.callback.Speedometer(100, 1000),mx.callback.ProgressBar],
#optimizer_params={'learning_rate':0.1},
                #batch_end_callback = 
                #[mx.callback.ProgressBar(train_iter.totalnum()),mx.callback.Speedometer(batch_num,train_iter.totalnum()-1)],
print train_iter.totalnum(),"batches"
model.fit(train_iter,
                optimizer=args.optimizer,
                epoch_end_callback=mx.callback.do_checkpoint('save/jetwcheck_'+args.network+'_'+str(start.date())),
                num_epoch=args.num_epochs)
#lenet_model.save_checkpoint(prefix='jeti1_1',epoch=10)

print datetime.datetime.now()-start