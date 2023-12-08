import os
import sys
cwd = os.getcwd()
sys.path.insert(0, '{}/python'.format(cwd))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import ROOT
import math

from Particle import Particle
from Utilities import PolarimetricVector, AcoplanarityAngle_PV

def main(args):
   cwd = os.getcwd()
   process=args.process
   df = pd.read_pickle("{}/samples/pickle/{}.pkl".format(cwd,process))
   print("Total number of events in file:", len(df))
   DM = args.decay_modes
   print(df['tau_decay_mode_1'].value_counts())
   if DM == "00":
      df = df[(df['tau_decay_mode_1'] == 0) & (df['tau_decay_mode_2'] == 0)]
   elif DM == "10":
      df = df[(df['tau_decay_mode_1'] == 1) & (df['tau_decay_mode_2'] == 0)]  
   elif DM == "01":
      df = df[(df['tau_decay_mode_1'] == 0) & (df['tau_decay_mode_2'] == 1)]
   elif DM == "11":
      df = df[(df['tau_decay_mode_1'] == 1) & (df['tau_decay_mode_2'] == 1)]
   print("Number of events of DMs {}: {}".format(DM,len(df)))
   df = df.head(100000)
   df.reset_index(drop=True,inplace = True)

   nevents = len(df)
   nu1 = [ROOT.TLorentzVector(df['gen_nu_px_1'].values[index],df['gen_nu_py_1'].values[index],df['gen_nu_pz_1'].values[index],math.sqrt(df['gen_nu_px_1'].values[index]**2+df['gen_nu_py_1'].values[index]**2+df['gen_nu_pz_1'].values[index]**2)) for index in range(nevents)]
   nu2 = [ROOT.TLorentzVector(df['gen_nu_px_2'].values[index],df['gen_nu_py_2'].values[index],df['gen_nu_pz_2'].values[index],math.sqrt(df['gen_nu_px_2'].values[index]**2+df['gen_nu_py_2'].values[index]**2+df['gen_nu_pz_2'].values[index]**2)) for index in range(nevents)]
   
   pi_1 = [Particle(ROOT.TLorentzVector(df['pi_px_1'].values[index],df['pi_py_1'].values[index],df['pi_pz_1'].values[index],df['pi_E_1'].values[index]),df['q_pi_1'].values[index]) for index in range(nevents)]
   pi_2 = [Particle(ROOT.TLorentzVector(df['pi_px_2'].values[index],df['pi_py_2'].values[index],df['pi_pz_2'].values[index],df['pi_E_2'].values[index]),df['q_pi_2'].values[index]) for index in range(nevents)]
   
   pi2_1 = [Particle(ROOT.TLorentzVector(df['pi2_px_1'].values[index],df['pi2_py_1'].values[index],df['pi2_pz_1'].values[index],df['pi2_E_1'].values[index]),df['q_pi2_1'].values[index]) for index in range(nevents)]
   pi2_2 = [Particle(ROOT.TLorentzVector(df['pi2_px_2'].values[index],df['pi2_py_2'].values[index],df['pi2_pz_2'].values[index],df['pi2_E_2'].values[index]),df['q_pi2_2'].values[index]) for index in range(nevents)]   

   pi3_1 = [Particle(ROOT.TLorentzVector(df['pi3_px_1'].values[index],df['pi3_py_1'].values[index],df['pi3_pz_1'].values[index],df['pi3_E_1'].values[index]),df['q_pi3_1'].values[index]) for index in range(nevents)]
   pi3_2 = [Particle(ROOT.TLorentzVector(df['pi3_px_2'].values[index],df['pi3_py_2'].values[index],df['pi3_pz_2'].values[index],df['pi3_E_2'].values[index]),df['q_pi3_2'].values[index]) for index in range(nevents)]  

   pi0_1 = [Particle(ROOT.TLorentzVector(df['pi0_px_1'].values[index],df['pi0_py_1'].values[index],df['pi0_pz_1'].values[index],df['pi0_E_1'].values[index]), 0.) for index in range(nevents)]
   pi0_2 = [Particle(ROOT.TLorentzVector(df['pi0_px_2'].values[index],df['pi0_py_2'].values[index],df['pi0_pz_2'].values[index],df['pi0_E_2'].values[index]), 0.) for index in range(nevents)]

   vis1=[pi_1[index].Fourvector + pi2_1[index].Fourvector + pi3_1[index].Fourvector +  pi0_1[index].Fourvector for index in range(nevents)]
   vis2=[pi_2[index].Fourvector + pi2_2[index].Fourvector + pi3_2[index].Fourvector +  pi0_2[index].Fourvector for index in range(nevents)]

   tau1 = [Particle(vis1[index] + nu1[index],df['q1'].values[index]) for index in range(nevents)]
   tau2 = [Particle(vis2[index] + nu2[index],df['q2'].values[index]) for index in range(nevents)] 

   PV1_true = [ROOT.TLorentzVector(df['PV1_x'].values[index],df['PV1_y'].values[index],df['PV1_z'].values[index],df['PV1_E'].values[index]) for index in range(nevents)]
   PV2_true = [ROOT.TLorentzVector(df['PV2_x'].values[index],df['PV2_y'].values[index],df['PV2_z'].values[index],df['PV2_E'].values[index]) for index in range(nevents)]

   PV1 = []
   PV2 = []
   for index in range(nevents):
      if (index % 10000 == 0): print(index)
      if DM == "00":
         PV1.append(PolarimetricVector([pi_1[index]], [], tau1[index]))
         PV2.append(PolarimetricVector([pi_2[index]], [], tau2[index]))  
      elif DM == "01":
         PV1.append(PolarimetricVector([pi_1[index]], [], tau1[index]))
         PV2.append(PolarimetricVector([pi_2[index]], [pi0_2[index]], tau2[index]))
      elif DM == "10":
         PV1.append(PolarimetricVector([pi_1[index]], [pi0_1[index]], tau1[index]))
         PV2.append(PolarimetricVector([pi_2[index]], [], tau2[index]))
      elif DM == "11":
         PV1.append(PolarimetricVector([pi_1[index]], [pi0_1[index]], tau1[index]))
         PV2.append(PolarimetricVector([pi_2[index]], [pi0_2[index]], tau2[index]))    


   df['PV_test0'] = [AcoplanarityAngle_PV(PV1[index],PV2[index],tau1[index].Fourvector,tau2[index].Fourvector) for index in range(nevents)]
   df['PV_test1'] = [AcoplanarityAngle_PV(PV1_true[index],PV2_true[index],tau1[index].Fourvector,tau2[index].Fourvector) for index in range(nevents)]

   bins = 30
   r = (0, 2 * np.pi)
   AcoPlot(bins,r,df['PV_Aco'],df,"Acolanarity angle distribution for $DM_{1}$ = 0 & $DM_{2}$ = 0 PV+PV","test.png")
   AcoPlot(bins,r,df['PV_test0'],df,"Acolanarity angle distribution for $DM_{1}$ = 0 & $DM_{2}$ = 0 PV+PV","test0.png")
   AcoPlot(bins,r,df['PV_test1'],df,"Acolanarity angle distribution for $DM_{1}$ = 0 & $DM_{2}$ = 0 PV+PV","test1.png")

def AcoPlot(bins,r,x,weights,title,savefig):
   fig, ax = plt.subplots()
   hist1, bin_edges, _ = ax.hist(x, bins=bins, range=r,  color='b', weights=weights["wt_cp_sm"].values, label='Scalar',histtype='step', lw=2)
   hist2, bin_edges, _ = ax.hist(x, bins=bins, range=r,  color='r', weights=weights["wt_cp_ps"].values, label='Pseudoscalar', histtype='step', lw=2)
   asymmetry_overall = (1/bins)*(np.sum(abs(hist1 - hist2) / (hist1 + hist2)))
   ax.text(0.5, 0.2, "Overall asymmetry: {:.3f}".format(asymmetry_overall), ha='center', va='center', transform=ax.transAxes)
   ax.set_title(title,fontsize=15)
   ax.set_xlabel('Acolanarity angle $\phi_A$')
   ax.set_ylabel('Events')
   ax.legend()
   plt.savefig(savefig)
   
if __name__ == "__main__":   
   parser = argparse.ArgumentParser(description="Load .pkl file and process it")
   parser.add_argument("-DMs", "--decay_modes", type=str, help="The decay mode combination to be processed")
   parser.add_argument("-p","--process",type=str, help="Process to loaded")
   args = parser.parse_args()
   main(args)
