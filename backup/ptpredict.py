from imiter import *
from rootiter import *
from ptiter import *
import mxnet as mx
import numpy as np
import random
import datetime
import argparse
from common import fit,data
from sklearn.model_selection import train_test_split
from importlib import import_module
#from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, roc_curve, average_precision_score
#python ptpredict --range "(0,1)" --date "2017-08-04"
parser=argparse.ArgumentParser()
parser.add_argument("--ran",default="range(0,1)",help='model range name')
parser.add_argument("--date",default="2017-08-04",help='model date name')
parser.add_argument("--gpus",default="0",help='the ports of gpus')
parser.add_argument("--batch_num",default=100,help='the number of each batch')
parser.add_argument("--entries",default=2,help='the number to take batches -1 is get all data')
parser.add_argument("--epoch",default=5,help='check point number')
parser.add_argument("--save",type=int,default=1,help='1 likelyhood 2 roc')
fit.add_fit_args(parser)
data.add_data_args(parser)

parser.set_defaults(
    network = None,
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

start=datetime.datetime.now()

batch_num=args.batch_num
entries=args.entries
ptb="range(0,1)"
#ptb="range(1,21)"
train_iter=ptiter('../jetsome-test.root',['data'],['softmax_label'],batch_size=batch_num,begin=0,end=5./7.,ptbin=args.ran,sli=0.5)
test_iter=ptiter('../jetsome-test.root',['data'],['softmax_label'],batch_size=batch_num,begin=5./7.,end=1.,ptbin=args.ran,sli=0.5)
#sli=0.3
if(args.network==None):
    sym,arg_params,aux_params=mx.model.load_checkpoint("save/jetpcheck_"+args.ran+"_"+args.date,args.epoch)
else:
    sym,arg_params,aux_params=mx.model.load_checkpoint("save/jetpcheck_"+args.network+"_"+args.ran+"_"+args.date,args.epoch)
#sym,arg_params,aux_params=mx.model.load_checkpoint("save/jetpcheck_range(1,21)_2017-08-03",7)

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
mod=mx.mod.Module(symbol=sym,context=fit.getcxt(args.gpus))
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
        else:
            q.append(b[i])
if(args.network==None):
  savename=args.date
else:
  savename=args.network+'_'+args.date
like=plt.figure(1)
if(args.save==1):
	plt.hist(q,bins=30,alpha=0.5,label='quark')
	plt.hist(g,bins=30,alpha=0.5,label='gluon')
	plt.legend(loc="upper center")
	plt.savefig('likelyhood_'+savename)
	print 'likelyhood_'+savename,"saved"

roc=plt.figure(2)
#plt.plot(groc,qroc,label=str(auc))
#plt.plot(x,y,label=str(xya))
#plt.legend(loc="lower left")
#plt.plot(qroc)
if(args.save==2):
	t_fpr,t_tpr, _ = roc_curve(a.label[0].asnumpy(),b)
	t_fnr = 1-t_fpr
	train_auc=np.around(auc(t_fpr,t_tpr),4)
	plt.plot(t_tpr,t_fnr,alpha=0.6,c="m",label="AUC = {}".format(train_auc),lw=2)
	plt.legend(loc='lower left')
	plt.savefig('rocp_'+savename)
	print 'rocp_'+savename,"saved"
	#plt.savefig(rocsave)
	print train_auc
print datetime.datetime.now()-start
