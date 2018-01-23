from imiter import *
from rootiter import *
from ptiter import *
import mxnet as mx
import random
import datetime
import argparse
import sys
from common import fit,data
from sklearn.model_selection import train_test_split
from importlib import import_module
#from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, roc_curve, average_precision_score
#python ptrun --network vgg --optimizer adagrad --num_epochs 10 --batch_sze 1000 --pt "range(0,1)"
parser=argparse.ArgumentParser()
#parser.add_argument("--network",default="vgg",help='the name of cnn architecture')
#parser.add_argument("--optimizer",default="adagrad",help='the optimizer at fitting')
#parser.add_argument("--num_epochs",type=int,default=10,help='the number of total epochs')
#parser.add_argument("--batch_size",type=int,default=100,help='the number of batch size')
parser.add_argument("--pt",default="range(0,1)",help='the bin number of pt range(0,21)')
parser.add_argument("--begin",type=float,default=0.,help='begin of training must begin<end')
parser.add_argument("--end",type=float,default=1.,help='end of training must begin<end')
parser.add_argument("--slicear",type=float,default=1.,help='slice of array for smaller training')
#parser.add_argument("--gpus",default=None,help='the ports of gpus')
fit.add_fit_args(parser)
data.add_data_args(parser)

parser.set_defaults(
    network = 'vgg',
    gpus=None,
    num_classes = 2,
    image_shape = '3,33,33',
    pad_size = 4,
    batch_size = 100,
    disp_batched = 99,
    num_epochs = 10,
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

batch_num=args.batch_size
ptb=args.pt

start=datetime.datetime.now()

train_iter=ptiter('../jetsome-test.root',['data'],['softmax_label'],batch_size=batch_num,begin=_beg,end=_mid,ptbin=ptb,sli=args.slicear)
test_iter=ptiter('../jetsome-test.root',['data'],['softmax_label'],batch_size=batch_num,begin=_mid,end=_end,ptbin=ptb,sli=args.slicear)

net = import_module('symbols.'+args.network)

sym=net.get_symbol((**vars(args))
print "vgg"



import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
# create a trainable module on GPU 0
print args.network
model = mx.mod.Module(symbol=sym, context=fit.getctx(args.gpus))

print train_iter.trainnum(),"batchs"
model.fit(train_iter,
                eval_data=test_iter,
                optimizer=args.optimizer,
                eval_metric='acc',
                batch_end_callback = 
                [mx.callback.ProgressBar(train_iter.totalnum()),mx.callback.Speedometer(batch_num,train_iter.totalnum()-1)],
                epoch_end_callback=mx.callback.do_checkpoint('save/jetpcheck_'+args.network+'_'+str(ptb)+'_'+str(start.date())),
                num_epoch=args.num_epochs)
#lenet_model.save_checkpoint(prefix='jeti1_1',epoch=10)

print datetime.datetime.now()-start
