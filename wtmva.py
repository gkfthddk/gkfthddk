import argparse
import ROOT
parser=argparse.ArgumentParser()
parser.add_argument("--save",type=str,default="exercise",help="save output file name")
parser.add_argument("--left",type=str,default="qq",help="signal train data")
parser.add_argument("--right",type=str,default="gg",help="background train data")
parser.add_argument("--teleft",type=str,default="qq100img",help="signal test data")
parser.add_argument("--teright",type=str,default="gg100img",help="background test data")
parser.add_argument("--ntrees",type=int,default=500,help="number of ntrees")
parser.add_argument("--rat",type=str,default="",help="rat for mixed")
args=parser.parse_args()
### Initialize TMVA and create a factory object and dataloader object
#ROOT.TMVA.Tools.Instance()

#Create a ROOT output file where TMVA will store ntuples, histograms, etc.
outputFilename = "{}.root".format(args.save)
fout = ROOT.TFile(outputFilename,"RECREATE")
 
factory = ROOT.TMVA.Factory("{}".format(args.save), fout, "V:!Silent:Color:DrawProgressBar:Transformations=I;P;G:AnalysisType=Classification")
# The values given to the Transformations parameter only affect testing and visualization, not the training (according to the manual).
(ROOT.TMVA.gConfig().GetVariablePlotting()).fMaxNumOfAllowedVariablesForScatterPlots = 9999
(ROOT.TMVA.gConfig().GetVariablePlotting()).fNbinsXOfROCCurve = 500;
rat=args.rat.split(",")
dataloader=[]
sigfile=[]
bakfile=[]
sig=[]
bak=[]

valsigfile=ROOT.TFile.Open("./data/val{}.root".format(args.teleft))
valsig=valsigfile.Get("image")
testsigfile=ROOT.TFile.Open("./data/test{}.root".format(args.teleft))
testsig=testsigfile.Get("image")

valbakfile=ROOT.TFile.Open("./data/val{}.root".format(args.teright))
valbak=valbakfile.Get("image")
testbakfile=ROOT.TFile.Open("./data/test{}.root".format(args.teright))
testbak=testbakfile.Get("image")

signalTrainWeight = 1.0;
signalTestWeight = 1.0;
backgroundTrainWeight = 1.0;
backgroundTestWeight = 1.0;
sigCut = ROOT.TCut("") # for example: ROOT.TCut("abs(var1) <0.5 && abs(var2 - 0.5) < 1")
bgCut = ROOT.TCut("")  
method1=[]
method2=[]
for i in range(len(rat)):
  rat[i]=eval(rat[i])
for i in range(len(rat)):
  dataloader.append(ROOT.TMVA.DataLoader("weakloader{}".format(str(int(rat[i]*100)))))
  dataloader[i].AddVariable("ptD",'F')
  dataloader[i].AddVariable("pt",'F')
  dataloader[i].AddVariable("axis1",'F')
  dataloader[i].AddVariable("axis2", 'F')
  dataloader[i].AddVariable("nmult", 'I')
  dataloader[i].AddVariable("cmult", 'I')
  
  sigfile.append(ROOT.TFile.Open("./data/train{}{}img.root".format(args.left,str(int(rat[i]*100)))))
  sig.append(sigfile[i].Get("image"))
  bakfile.append(ROOT.TFile.Open("./data/train{}{}img.root".format(args.right,str(int(rat[i]*100)))))
  bak.append(bakfile[i].Get("image"))

  dataloader[i].AddSignalTree( sig[i], signalTrainWeight, "Training");
  dataloader[i].AddSignalTree( valsig, signalTestWeight, "Test");
  dataloader[i].AddSignalTree( testsig, signalTestWeight, "Test");
  dataloader[i].AddBackgroundTree( bak[i], backgroundTrainWeight, "Training");
  dataloader[i].AddBackgroundTree( valbak, backgroundTestWeight, "Test");
  dataloader[i].AddBackgroundTree( testbak, backgroundTestWeight, "Test");

  dataloader[i].PrepareTrainingAndTestTree(sigCut,bgCut,"nTrain_Signal=0:nTrain_Background=0:SplitMode=Random:NormMode=NumEvents:VerboseLevel=Verbose")

  method1.append(factory.BookMethod( dataloader[i], ROOT.TMVA.Types.kBDT, "BDT","!H:!V:NTrees={}:CreateMVAPdfs:MinNodeSize=5%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning".format(args.ntrees)))
  method2.append(factory.BookMethod( dataloader[i], ROOT.TMVA.Types.kBDT, "BDTG","!H:!V:NTrees={}:CreateMVAPdfs:MinNodeSize=5%:MaxDepth=3:BoostType=Grad:Shrinkage=0.1:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning".format(args.ntrees)))
 

factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()

fout.Close()

ROOT.TMVA.TMVAGui(outputFilename )
