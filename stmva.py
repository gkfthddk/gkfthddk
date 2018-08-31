import argparse
import ROOT
parser=argparse.ArgumentParser()
parser.add_argument("--save",type=str,default="exercise",help="save output file name")
parser.add_argument("--left",type=str,default="ppqq_img",help="signal train data")
parser.add_argument("--right",type=str,default="ppgg_img",help="background train data")
parser.add_argument("--dlinput",type=str,default="qqgg",help="dl input file name")
parser.add_argument("--teleft",type=str,default="ppzq_img",help="signal test data")
parser.add_argument("--teright",type=str,default="ppzg_img",help="background test data")
parser.add_argument("--ntrees",type=int,default=500,help="number of ntrees")
args=parser.parse_args()
### Initialize TMVA and create a factory object and dataloader object
#ROOT.TMVA.Tools.Instance()

#Create a ROOT output file where TMVA will store ntuples, histograms, etc.
outputFilename = "{}.root".format(args.save)
fout = ROOT.TFile(outputFilename,"RECREATE")
 
factory = ROOT.TMVA.Factory("{}".format(args.save), fout, "V:!Silent:Color:DrawProgressBar:Transformations=I;P;G:AnalysisType=Classification")
# The values given to the Transformations parameter only affect testing and visualization, not the training (according to the manual).

dataloader = ROOT.TMVA.DataLoader("dizloader")
dataloader2 = ROOT.TMVA.DataLoader("zdiloader")
dataloader3 = ROOT.TMVA.DataLoader("zqggloader")
dataloader4 = ROOT.TMVA.DataLoader("qqzgloader")

# If you wish to modify default settings
# please check "src/Config.h" to see all available global options
# or https://root.cern.ch/doc/v608/classTMVA_1_1Config_1_1VariablePlotting.html
(ROOT.TMVA.gConfig().GetVariablePlotting()).fMaxNumOfAllowedVariablesForScatterPlots = 9999
(ROOT.TMVA.gConfig().GetVariablePlotting()).fNbinsXOfROCCurve = 500;

### Registration of variables
dataloader.AddVariable("ptD",'F')
dataloader.AddVariable("pt",'F')
dataloader.AddVariable("axis1",'F')
dataloader.AddVariable("axis2", 'F')
dataloader.AddVariable("nmult", 'I')
dataloader.AddVariable("cmult", 'I')
dataloader2.AddVariable("ptD",'F')
dataloader2.AddVariable("pt",'F')
dataloader2.AddVariable("axis1",'F')
dataloader2.AddVariable("axis2", 'F')
dataloader2.AddVariable("nmult", 'I')
dataloader2.AddVariable("cmult", 'I')
dataloader3.AddVariable("ptD",'F')
dataloader3.AddVariable("pt",'F')
dataloader3.AddVariable("axis1",'F')
dataloader3.AddVariable("axis2", 'F')
dataloader3.AddVariable("nmult", 'I')
dataloader3.AddVariable("cmult", 'I')
dataloader4.AddVariable("ptD",'F')
dataloader4.AddVariable("pt",'F')
dataloader4.AddVariable("axis1",'F')
dataloader4.AddVariable("axis2", 'F')
dataloader4.AddVariable("nmult", 'I')
dataloader4.AddVariable("cmult", 'I')

#for i in range(9,79,10):
#  dataloader2.AddVariable("RKF["+str(i)+"]", 'F' );

### Registration of trees
Sigfname = "./data/{}.root".format(args.left);
Bkgfname = "./data/{}.root".format(args.right);
STestfname = "./data/{}.root".format(args.teleft);
BTestfname = "./data/{}.root".format(args.teright);

Sinput = ROOT.TFile.Open(Sigfname);
Binput = ROOT.TFile.Open(Bkgfname);

STestinput = ROOT.TFile.Open(STestfname);
BTestinput = ROOT.TFile.Open(BTestfname);

signal     = Sinput.Get("image");
background = Binput.Get("image");
signaltest     = STestinput.Get("image");
backgroundtest     = BTestinput.Get("image");
#global event weights per tree (see below for setting event-wise weights)
signalTrainWeight = 1.0;
signalTestWeight = 1.0;
backgroundTrainWeight = 1.0;
backgroundTestWeight = 1.0;
#Set individual event weights (the variables must exist in the original TTree)
# -  for signal    : `dataloader->SetSignalWeightExpression    ("weight1*weight2");`
# -  for background: `dataloader->SetBackgroundWeightExpression("weight1*weight2");`

#You can add an arbitrary number of signal or background trees
#dataloader.AddSignalTree    ( signal,     signalTrainWeight     );
#dataloader.AddBackgroundTree( background, backgroundTrainWeight );

#To give different trees for training and testing, do as follows:      
dataloader.AddSignalTree( signal, signalTrainWeight, "Training" );
dataloader.AddSignalTree( signaltest,     signalTestWeight,  "Test" );
dataloader.AddBackgroundTree( background, backgroundTrainWeight, "Training" );
dataloader.AddBackgroundTree( backgroundtest,     backgroundTestWeight,  "Test" );
dataloader2.AddSignalTree( signaltest, signalTestWeight, "Training" );
dataloader2.AddSignalTree( signal,     signalTrainWeight,  "Test" );
dataloader2.AddBackgroundTree( backgroundtest, backgroundTestWeight, "Training" );
dataloader2.AddBackgroundTree( background,     backgroundTrainWeight,  "Test" );
dataloader3.AddSignalTree( signaltest, signalTestWeight, "Training" );
dataloader3.AddSignalTree( signal,     signalTrainWeight,  "Test" );
dataloader3.AddBackgroundTree( background, backgroundTrainWeight, "Training" );
dataloader3.AddBackgroundTree( backgroundtest,     backgroundTestWeight,  "Test" );
dataloader4.AddSignalTree( signal, signalTrainWeight, "Training" );
dataloader4.AddSignalTree( signaltest,     signalTestWeight,  "Test" );
dataloader4.AddBackgroundTree( backgroundtest, backgroundTestWeight, "Training" );
dataloader4.AddBackgroundTree( background,     backgroundTrainWeight,  "Test" );


### Preparation
# cuts defining the signal and background sample
sigCut = ROOT.TCut("") # for example: ROOT.TCut("abs(var1) <0.5 && abs(var2 - 0.5) < 1")
bgCut = ROOT.TCut("")  

#Tell the dataloader how to use the training and testing events
#If no numbers of events are given, half of the events in the tree are used for training, and the other half for testing:
dataloader.PrepareTrainingAndTestTree(sigCut,bgCut,"SplitMode=Random:NormMode=NumEvents:VerboseLevel=Verbose")
dataloader2.PrepareTrainingAndTestTree(sigCut,bgCut,"SplitMode=Random:NormMode=NumEvents:VerboseLevel=Verbose")
dataloader3.PrepareTrainingAndTestTree(sigCut,bgCut,"SplitMode=Random:NormMode=NumEvents:VerboseLevel=Verbose")
dataloader4.PrepareTrainingAndTestTree(sigCut,bgCut,"SplitMode=Random:NormMode=NumEvents:VerboseLevel=Verbose")

### Configure classifier (BDT)
method1 = factory.BookMethod( dataloader, ROOT.TMVA.Types.kBDT, "BDT","!H:!V:NTrees={}:CreateMVAPdfs:MinNodeSize=5%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning".format(args.ntrees))
method2 = factory.BookMethod( dataloader, ROOT.TMVA.Types.kBDT, "BDTG","!H:!V:NTrees={}:CreateMVAPdfs:MinNodeSize=5%:MaxDepth=3:BoostType=Grad:Shrinkage=0.1:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning".format(args.ntrees))
method21 = factory.BookMethod( dataloader2, ROOT.TMVA.Types.kBDT, "BDT","!H:!V:NTrees={}:CreateMVAPdfs:MinNodeSize=5%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning".format(args.ntrees))
method22 = factory.BookMethod( dataloader2, ROOT.TMVA.Types.kBDT, "BDTG","!H:!V:NTrees={}:CreateMVAPdfs:MinNodeSize=5%:MaxDepth=3:BoostType=Grad:Shrinkage=0.1:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning".format(args.ntrees))
method31 = factory.BookMethod( dataloader3, ROOT.TMVA.Types.kBDT, "BDT","!H:!V:NTrees={}:CreateMVAPdfs:MinNodeSize=5%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning".format(args.ntrees))
method32 = factory.BookMethod( dataloader3, ROOT.TMVA.Types.kBDT, "BDTG","!H:!V:NTrees={}:CreateMVAPdfs:MinNodeSize=5%:MaxDepth=3:BoostType=Grad:Shrinkage=0.1:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning".format(args.ntrees))
method41 = factory.BookMethod( dataloader4, ROOT.TMVA.Types.kBDT, "BDT","!H:!V:NTrees={}:CreateMVAPdfs:MinNodeSize=5%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning".format(args.ntrees))
method42 = factory.BookMethod( dataloader4, ROOT.TMVA.Types.kBDT, "BDTG","!H:!V:NTrees={}:CreateMVAPdfs:MinNodeSize=5%:MaxDepth=3:BoostType=Grad:Shrinkage=0.1:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning".format(args.ntrees))
 

factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()

fout.Close()

ROOT.TMVA.TMVAGui(outputFilename )
