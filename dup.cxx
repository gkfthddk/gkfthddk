#include "TFile.h"
#include "TTree.h"
#include<iostream>
#include<string>
#include<vector>
#include<cmath>

/*toim : toim.cxx
	g++ -std=c++0x toim.cxx -o toim `root-config --cflags --libs`*/


int main() {
  const int arnum=16;
  const float maxx=0.4;
  const float maxy=0.4;

  auto rf = new TFile("../jet_pythia_100.root");
  auto jet = (TTree*) rf->Get("jetAnalyser/jetAnalyser");
  int batch_size = -1;

  int partonId;
  jet->SetBranchAddress("partonId", &partonId);
  float pt,eta;
  int nJets,nMatchedJets,nGenJets;
  jet->SetBranchAddress("pt",&pt);
  jet->SetBranchAddress("eta",&eta);
  jet->SetBranchAddress("nJets",&nJets);
  jet->SetBranchAddress("nMatchedJets",&nMatchedJets);
  jet->SetBranchAddress("nGenJets",&nGenJets);
  std::vector<float> *dau_pt = 0;
  jet->SetBranchAddress("dau_pt", &dau_pt);
  std::vector<float> *dau_deta = 0;
  jet->SetBranchAddress("dau_deta", &dau_deta);
  std::vector<float> *dau_dphi = 0;
  jet->SetBranchAddress("dau_dphi", &dau_dphi);
  std::vector<int> *dau_charge = 0;
  jet->SetBranchAddress("dau_charge", &dau_charge);

  auto outf = new TFile("jetimgnumtest.root", "recreate");
  auto t = new TTree("image", "image");
  auto quark = new TTree("qimage", "qimage");
  auto gluon = new TTree("gimage", "gimage");
  t->SetDirectory(outf);
  quark->SetDirectory(outf);
  gluon->SetDirectory(outf);
  const int ch_size = (2*arnum+1)*(2*arnum+1);
  const int stride = (2*arnum+1);
  const int img_size = 3*(2*arnum+1)*(2*arnum+1);
  unsigned char arr[img_size];
  unsigned char label;
  t->Branch("image", arr, (std::string("image[") + std::to_string(img_size) + "]/b").c_str());
  t->Branch("label", &label, "label/b");
  t->Branch("nJets", &nJets, "MatchedJets/I");
  t->Branch("nMatchedJets", &nMatchedJets, "nMatchedJets/I");
  t->Branch("nGenJets", &nGenJets, "GenJets/I");
  t->Branch("pt", &pt, "pt/F");
  t->Branch("eta", &eta, "eta/F");
  quark->Branch("image", arr, (std::string("image[") + std::to_string(img_size) + "]/b").c_str());
  quark->Branch("label", &label, "label/b");
  quark->Branch("nJets", &nJets, "MatchedJets/I");
  quark->Branch("nMatchedJets", &nMatchedJets, "nMatchedJets/I");
  quark->Branch("nGenJets", &nGenJets, "GenJets/I");
  quark->Branch("pt", &pt, "pt/F");
  quark->Branch("eta", &eta, "eta/F");
  gluon->Branch("image", arr, (std::string("image[") + std::to_string(img_size) + "]/b").c_str());
  gluon->Branch("label", &label, "label/b");
  gluon->Branch("nJets", &nJets, "MatchedJets/I");
  gluon->Branch("nMatchedJets", &nMatchedJets, "nMatchedJets/I");
  gluon->Branch("nGenJets", &nGenJets, "GenJets/I");
  gluon->Branch("pt", &pt, "pt/F");
  gluon->Branch("eta", &eta, "eta/F");


  double bx=2*maxx/(2*arnum+1);
  double by=2*maxy/(2*arnum+1);

  if (batch_size == -1)
    batch_size = jet->GetEntries();
  for (int i = 0; i < batch_size; ++i) {
    if (i % 10000 == 0)
      std::cout << i <<  '/' << batch_size << std::endl;
    jet->GetEntry(i);
    if (partonId==0)
      continue;
    if(pt<100 || nMatchedJets!=2 || eta>2.4){
      continue;
    }

    double r=0, g=0, b=0;
    double palet[3][2*arnum+1][2*arnum+1];
    for (int ii = 0; ii < 3; ++ii)
      for (int jj = 0; jj < (2*arnum+1); ++jj)
	for (int kk = 0; kk < (2*arnum+1); ++kk)
	  palet[ii][jj][kk] = 0;
    for (int j = 0; j < dau_pt->size(); ++j) {
      int x = std::floor(((dau_deta->at(j)/bx)+0.5)+arnum);
      if ((x < 0) or (x > (2*arnum)))
	continue;
      int y = std::floor(((dau_dphi->at(j)/by)+0.5)+arnum);
      if ((y < 0) or (y > (2*arnum)))
	continue;
      double dpt = dau_pt->at(j);
      if (dau_charge->at(j) == 0) {
	palet[1][x][y] += dpt;
	if (palet[1][x][y] > g)
	  g = palet[1][x][y];
      } else {
	palet[0][x][y] += dpt;
	palet[2][x][y] += 1;
	if (palet[0][x][y]>r)
	  r=palet[0][x][y];
	if (palet[2][x][y]>b)
	  b=palet[2][x][y];
      }
    }

    for (int ii = 0; ii < (2*arnum+1); ++ii) {
      for (int jj = 0; jj < (2*arnum+1); ++jj) {
	if (r!=0)
	  palet[0][ii][jj]=255*palet[0][ii][jj]/r;
	if (g!=0)
	  palet[1][ii][jj]=255*palet[1][ii][jj]/g;
	if (b!=0)
	  palet[2][ii][jj]=255*palet[2][ii][jj]/b;

	// if (palet[0][ii][jj] < 0) palet[0][ii][jj] = 0;
	// if (palet[1][ii][jj] < 0) palet[1][ii][jj] = 0;
	// if (palet[2][ii][jj] < 0) palet[2][ii][jj] = 0;
	
	arr[0*ch_size + ii*stride + jj] = int(palet[0][ii][jj]);
	arr[1*ch_size + ii*stride + jj] = int(palet[1][ii][jj]);
	arr[2*ch_size + ii*stride + jj] = int(palet[2][ii][jj]);
      }
    }
    
    if (partonId==21){
      label = 1;
      gluon->SetDirectory(outf);
      gluon->Fill();}
    else{
      label = 0;
      quark->SetDirectory(outf);
      quark->Fill();}

    
    t->SetDirectory(outf);
    t->Fill();
  }
  
  t->SetDirectory(outf);
  t->Write();
  quark->SetDirectory(outf);
  quark->Write();
  gluon->SetDirectory(outf);
  gluon->Write();
  outf->Close();
}
