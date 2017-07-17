import mxnet as mx
import ROOT as rt
import numpy as np
import math
import datetime
import sys
start=datetime.datetime.now()
buftime=datetime.datetime.now()
print("< - - - - - - - - - - >")
entries=10000000
for ent in range(entries):
    if(ent%int(entries/999)==0 and ent!=0):
        sys.stdout.write("\r%0.2f%%\t time remain %s\t" %
        (float(100.*ent/entries),str(int(1000.-1000.*ent/entries)*(datetime.datetime.now()-buftime))))
	sys.stdout.flush()
        buftime=datetime.datetime.now()
	#print datetime.datetime.now()
    #if ent>1000:
    #    break
#labels=np.array(labels)
#jetset=np.array(jetset,dtype=np.uint8)
#data=jetset.reshape((len(jetset),3*33*33))
print("<>")
print(datetime.datetime.now()-start)
