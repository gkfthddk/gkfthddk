from imiter import *
from rootiter import *
import mxnet as mx
import random
import datetime
import argparse
import sys
from sklearn.model_selection import train_test_split
#from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, roc_curve, average_precision_score
parser=argparse.ArgumentParser()
parser.add_argument("--batch_size",type=int,default=100,help='the number of batch size')
parser.add_argument("--num_epochs",type=int,default=10,help='the number of total epochs')
parser.add_argument("--gpus",default="1",help='the ports of gpus')
parser.add_argument("--begin",type=float,default=0.,help='begin of training must begin<end')
parser.add_argument("--end",type=float,default=1.,help='end of training must begin<end')
parser.add_argument("--optimizer",default="adagrad",help='the optimizer at fitting')
parser.add_argument("--network",default="vgg",help='the network at fitting')

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
train_iter=imiter('../jetsome-test.root',['data'],['softmax_label'],batch_size=batch_num,begin=_beg,end=_mid)
test_iter=imiter('../jetsome-test.root',['data'],['softmax_label'],batch_size=batch_num,begin=_mid,end=_end)


# __init__(self,data_path,data_names,label_names,batch_size=100,begin=0.0,end=1.0,endcut=1,arnum=16,maxx=0.4,maxy=0.4)
#for batch in train_iter:
#    pass
#data_train=data1[:int(len(data1)*0.7)]
#data_test=data1[int(len(data1)*0.7):]
#label_train=labels[:int(len(labels)*0.7)]
#label_test=labels[int(len(labels)*0.7):]

data = mx.sym.var('data')
# first conv layer
conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
tanh1 = mx.sym.Activation(data=conv1, act_type="relu")
pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
# second conv layer
conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)
tanh2 = mx.sym.Activation(data=conv2, act_type="relu")
pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
# first fullc layer
flatten = mx.sym.flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 = mx.sym.Activation(data=fc1, act_type="relu")
# second fullc
fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=2)
# softmax loss
lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
#mx.viz.plot_network(lenet)


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
if("vgg"==args.network):
    print("true")
lenet_model = mx.mod.Module(symbol=eval(args.network), context=[mx.gpu(0)])
print "gpu pass"
# train with the same 
"""
batch_end_callback = mx.callback.Speedometer(batch_size, 1000),
optimizer_params={'learning_rate':0.1},
"""
#optimizer_params={'learning_rate':0.5,'beta1':0.1,'beta2':0.111},
#batch_end_callback = [mx.callback.Speedometer(100, 1000),mx.callback.ProgressBar],
#optimizer_params={'learning_rate':0.1},
print train_iter.trainnum(),"batches"
lenet_model.fit(train_iter,
                eval_data=test_iter,
                optimizer=args.optimizer,
                eval_metric='acc',
                batch_end_callback = 
                [mx.callback.ProgressBar(train_iter.totalnum()),mx.callback.Speedometer(batch_num,train_iter.totalnum()-1)],
                epoch_end_callback=mx.callback.do_checkpoint('save/jeticheck_'+str(start.date())),
                num_epoch=args.num_epochs)
#lenet_model.save_checkpoint(prefix='jeti1_1',epoch=10)

print datetime.datetime.now()-start
