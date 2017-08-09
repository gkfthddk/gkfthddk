from imiter import *
from rootiter import *
from ptiter import *
import mxnet as mx
import numpy as np
import random
import datetime
import argparse
from sklearn.model_selection import train_test_split
#from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, roc_curve, average_precision_score
#python ptpredict --range "(0,1)" --date "2017-08-04"
parser=argparse.ArgumentParser()
parser.add_argument("--ran",default="range(0,1)",help='model range name')
parser.add_argument("--date",default="2017-08-04",help='model date name')
parser.add_argument("--gpus",default="0",help='the ports of gpus')

args=parser.parse_args()
print args

start=datetime.datetime.now()

batch_num=100
entries=2-1
ptb="range(0,1)"
#ptb="range(1,21)"
train_iter=ptiter('../jetsome-test.root',['data'],['softmax_label'],batch_size=batch_num,begin=0,end=5./7.,ptbin=args.ran,sli=0.5)
test_iter=ptiter('../jetsome-test.root',['data'],['softmax_label'],batch_size=batch_num,begin=5./7.,end=1.,ptbin=args.ran,sli=0.5)
#sli=0.3

sym,arg_params,aux_params=mx.model.load_checkpoint("save/jetpcheck_"+args.ran+"_"+args.date,6)
#sym,arg_params,aux_params=mx.model.load_checkpoint("save/jetpcheck_range(1,21)_2017-08-03",7)

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
vggnet=mx.sym.SoftmaxOutput(data=fc3, name='softmax')
#mx.viz.plot_network(vggnet,shape={"data":(1,3,33,33)})
print "vgg"



import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
# create a trainable module on GPU 0
#lenet_model = mx.mod.Module(symbol=vggnet, context=mx.gpu(1))
# train with the same 
cxt=[]
for i in args.gpus.split(","):
    cxt.append(mx.gpu(eval(i)))
mod=mx.mod.Module(symbol=sym,context=cxt)
print "mod"
mod.bind(data_shapes=test_iter.provide_data,label_shapes=test_iter.provide_label)
print "bind"
mod.init_params()
print "init"
mod.set_params(arg_params,aux_params)
print "loaded"
q=[]
g=[]
x=[]
y=[]
ent=0
qgauc=0
xya=0
groc=[]
qroc=[]
qm=0
gm=0
start=datetime.datetime.now()
buftime=datetime.datetime.now()
print test_iter.totalnum(),entries

from sklearn.metrics import roc_auc_score, auc,precision_recall_curve,roc_curve,average_precision_score

for i in range(100):
    x.append(i/100.)
    y.append(1.-i/100.)
    groc.append(0)
    qroc.append(0)
if(entries==-1):
    entries=test_iter.totalnum()
for j in range(entries):
    a=test_iter.next()
    mod.forward(a)
    b=mod.get_outputs()[0].asnumpy()[:,1]
    for i in range(batch_num):
        #sys.stdout.write("\r%0.2f"%
        #                (float(100.*ent/entries)))
        #sys.stdout.flush()
        #buftime=datetime.datetime.now()
        #ent+=1
        if (a.label[0].asnumpy()[i]==1):
            g.append(b[i])
            groc[int(b[i]*100)]+=1.
        else:
            q.append(b[i])
            qroc[int(b[i]*100)]+=1.
t_fpr,t_tpr, _ = roc_curve(a.label[0].asnumpy(),b)
t_fnr = 1-t_fpr
train_auc=np.around(auc(t_fpr,t_tpr),4)
for i in range(1,100):
    groc[i]+=groc[i-1]
    qroc[i]+=qroc[i-1]
for i in range(0,100):
    groc[i]=1-groc[i]/groc[99]
    qroc[i]=qroc[i]/qroc[99]
    qgauc+=qroc[i]/100.
    xya+=y[i]/100.
like=plt.figure(1)
plt.hist(q,bins=30,alpha=0.5,label='quark')
plt.hist(g,bins=30,alpha=0.5,label='gluon')
plt.legend(loc="upper center")
plt.savefig(str('likelyhoodp'+args.ran))

#roc=plt.figure(2)
#plt.plot(groc,qroc,label=str(auc))
#plt.plot(x,y,label=str(xya))
#plt.legend(loc="lower left")
#plt.plot(qroc)
"""plt.plot(t_tpr,t_fnr,alpha=0.6,c="m",label="Training AUC = {}".format(train_auc),lw=2)
plt.legend(loc='lower left')
plt.savefig('rocscp'+args.ran)"""
print train_auc
print datetime.datetime.now()-start
