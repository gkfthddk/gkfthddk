import mxnet as mx
import ROOT as rt
import numpy as np
import math
import datetime
import sys
start=datetime.datetime.now()
buftime=datetime.datetime.now()
print("< - - - - - - - - - - >")
jetset=[]
labels=[]
maxx=0.4
maxy=0.4                  
arnum=16
dx=0
bx=2*maxx/(2*arnum+1)
by=2*maxy/(2*arnum+1)
read=rt.TFile('jet2.root')
train=mx.recordio.MXRecordIO('tmpa2dd.rec','w')
laben=mx.recordio.MXRecordIO('laba2dd.rec','w')
jet=rt.gDirectory.Get('jetAnalyser/jetAnalyser')
entries=jet.GetEntriesFast()
for ent in range(entries):
    if(ent%int(entries/999)==0 and ent!=0):
        sys.stdout.write("\r%0.2f%%\t time remain %s\t" %
        (float(100.*ent/entries),str(int(1000.-1000.*ent/entries)*(datetime.datetime.now()-buftime))))
	sys.stdout.flush()
        buftime=datetime.datetime.now()
	#print datetime.datetime.now()
    #if ent>1000:
    #    break
    jet.GetEntry(ent)
    #if jet.pt<300 or jet.pt>400:
        #continue
    if jet.partonId==0:
        continue
    palet=np.zeros(shape=(3,2*arnum+1,2*arnum+1),dtype=np.float16)
    ddd=[]
    r=0
    g=0
    b=0
    for i in range(len(jet.dau_pt)):
        x=int(math.floor((jet.dau_deta[i]/bx)+0.5)+arnum)
        if x<0 or x>2*arnum:
            continue
        y=int(math.floor((jet.dau_dphi[i]/by)+0.5)+arnum)
        if y<0 or y>2*arnum:
            continue
        pt=jet.dau_pt[i]
        if jet.dau_charge[i]==0:
            palet[1][x][y]+=pt

            if palet[1][x][y]>g:
                g=palet[1][x][y]
        else:
            palet[0][x][y]+=pt
            palet[2][x][y]+=1

            if palet[0][x][y]>r:
                r=palet[0][x][y]
            if palet[2][x][y]>b:
                b=palet[2][x][y]
            #pass
    #record.write(palet)
    for i in range(2*arnum+1):
        for j in range(2*arnum+1):
            if r!=0:
                palet[0][i][j]=255*palet[0][i][j]/r
            if g!=0:
                palet[1][i][j]=255*palet[1][i][j]/g
            if b!=0:
                palet[2][i][j]=255*palet[2][i][j]/b
            if palet[0][i][j]!=0:
                ddd.append([i,j,0,int(palet[0][i][j])])
            if palet[1][i][j]!=0:
                ddd.append([i,j,1,int(palet[1][i][j])])
            if palet[2][i][j]!=0:
                ddd.append([i,j,2,int(palet[2][i][j])])
    
    jetset.append(ddd)
    if jet.partonId==21:
        labels.append(1)
    else:
        labels.append(0)
    if (ent%1000==0):
        #jetset=np.array(jetset,dtype=np.uint8)
        for l in range(len(jetset)):
            #record.write_idx(dx,str(jetset[l].tolist()))
            train.write(str(jetset[l]))
            dx+=1
        for m in range(len(labels)):
            laben.write(str(labels[l]))
        labels=[]
        jetset=[]
[r,g,b]=[0,0,0]
[gr,gg,gb]=[0,0,0]
[qr,qg,qb]=[0,0,0]
                
train.close()
laben.close()
#labels=np.array(labels)
#jetset=np.array(jetset,dtype=np.uint8)
#data=jetset.reshape((len(jetset),3*33*33))
print("<>")
print(datetime.datetime.now()-start)
