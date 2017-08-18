import mxnet as mx
import random
import datetime
import sys
from importlib import import_module



def getsym(network,num_classes):
    net=import_module('symbols.'+network)
    sym=net.get_symbol(num_classes)
    return sym

def getctx(gpus,cpus):
    if(gpus==None):
        cxt=mx.cpu()
	if(cpus!=None):
            ctx=[]
            for i in cpus.split(","):
                ctx.append(mx.cpu(eval(i)))
    else:
        cxt=[]
        for i in gpus.split(","):
            cxt.append(mx.gpu(eval(i)))
    return cxt


