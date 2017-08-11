from imiter import *
from rootiter import *
from ptiter import *
import mxnet as mx
import random
import datetime
import argparse
import sys
from sklearn.model_selection import train_test_split
from importlib import import_module
#from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, roc_curve, average_precision_score
#python ptrun --network vgg --optimizer adagrad --num_epochs 10 --batch_sze 1000 --pt "range(0,1)"
parser=argparse.ArgumentParser()
parser.add_argument("--network",default="vgg",help='the name of cnn architecture')
parser.add_argument("--optimizer",default="adagrad",help='the optimizer at fitting')
parser.add_argument("--num_epochs",type=int,default=10,help='the number of total epochs')
parser.add_argument("--batch_size",type=int,default=100,help='the number of batch size')
parser.add_argument("--pt",default="range(0,1)",help='the bin number of pt range(0,21)')
parser.add_argument("--begin",type=float,default=0.,help='begin of training must begin<end')
parser.add_argument("--end",type=float,default=1.,help='end of training must begin<end')
parser.add_argument("--slicear",type=float,default=1.,help='slice of array for smaller training')
parser.add_argument("--gpus",default="0",help='the ports of gpus')
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

sym=net.get_symbol(2)

data=mx.sym.var('data')
conv1 = mx.sym.Convolution(data=data, kernel=(3,3),pad=(1,1), num_filter=64)
relu1 = mx.sym.Activation(data=conv1, act_type="relu")
pool1 = mx.sym.Pooling(data=relu1, pool_type="max", kernel=(2,2), stride=(2,2))
conv2 = mx.sym.Convolution(data=pool1, kernel=(3,3),pad=(1,1), num_filter=128)
relu2 = mx.sym.Activation(data=conv2, act_type="relu")
pool2 = mx.sym.Pooling(data=relu2, pool_type="max", kernel=(2,2), stride=(2,2))
conv3 = mx.sym.Convolution(data=pool2, kernel=(3,3),pad=(1,1), num_filter=256)
relu3 = mx.sym.Activation(data=conv3, act_type="relu")
conv4 = mx.sym.Convolution(data=relu3, kernel=(3,3),pad=(1,1), num_filter=256)
relu4 = mx.sym.Activation(data=conv4, act_type="relu")
pool3 = mx.sym.Pooling(data=relu4, pool_type="max", kernel=(2,2), stride=(2,2))
flatten = mx.sym.flatten(data=pool3)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096)
relu9 = mx.sym.Activation(data=fc1, act_type="relu")
drop1=mx.sym.Dropout(data=relu9)
fc2 = mx.symbol.FullyConnected(data=drop1, num_hidden=4096)
relu10 = mx.sym.Activation(data=fc2, act_type="relu")
drop2=mx.sym.Dropout(data=relu10)
fc3 = mx.symbol.FullyConnected(data=drop2, num_hidden=2)
vgg=mx.sym.SoftmaxOutput(data=fc3, name='softmax')
#mx.viz.plot_network(vggnet,shape={"data":(1,3,33,33)})
print "vgg"



import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
# create a trainable module on GPU 0
cxt=[]
for i in args.gpus.split(","):
    cxt.append(mx.gpu(eval(i)))
print args.network
lenet_model = mx.mod.Module(symbol=sym, context=cxt)

print train_iter.trainnum(),"batchs"
lenet_model.fit(train_iter,
                eval_data=test_iter,
                optimizer=args.optimizer,
                eval_metric='acc',
                batch_end_callback = 
                [mx.callback.ProgressBar(train_iter.totalnum()),mx.callback.Speedometer(batch_num,train_iter.totalnum()-1)],
                epoch_end_callback=mx.callback.do_checkpoint('save/jetpcheck_'+str(ptb)+'_'+str(start.date())),
                num_epoch=args.num_epochs)
#lenet_model.save_checkpoint(prefix='jeti1_1',epoch=10)

print datetime.datetime.now()-start
