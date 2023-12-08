import os
import sys
directory = os.getcwd()
sys.path.append(directory+'/python/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import ROOT
import math
import pickle

from Particle import Particle
from Utilities import PolarimetricVector, AcoplanarityAngle_PV, AcoplanarityAngle
from plotting import PlotHistograms, AcoPlot

def main(args):
   cwd = os.getcwd()
   path=args.path
   DM = args.DM

   file_path = path + 'info.pkl'
   with open(file_path, 'rb') as file:
      dictionary = pickle.load(file)   
   
   inputs = dictionary['inputs']
   cat = dictionary['cat']
   targets_1 = dictionary['targets_1']
   targets_2 = dictionary['targets_2'] 
   predictions = dictionary['predictions']
   cpweights = dictionary['cpweights']
   target_1_means = dictionary['target_1_means']
   target_1_stds = dictionary['target_1_stds']
   target_2_means = dictionary['target_2_means']
   target_2_stds = dictionary['target_2_stds']   
   
   cat['tau_decay_mode_1'] = cat['tau_decay_mode_1'].astype(int)
   cat['tau_decay_mode_2'] = cat['tau_decay_mode_2'].astype(int)   
   
   print("Total Number of events", len(inputs))
   
   if DM == "00":
      mask = (cat['tau_decay_mode_1'] == 0) & (cat['tau_decay_mode_2'] == 0)
   elif DM == "10" or DM == "01":
      mask = ((cat['tau_decay_mode_1'] == 1) & (cat['tau_decay_mode_2'] == 0)) | ((cat['tau_decay_mode_1'] == 0) & (cat['tau_decay_mode_2'] == 1))
   else:
      mask = (cat['tau_decay_mode_1'] == 1) & (cat['tau_decay_mode_2'] == 1)

   targets_1 = (targets_1 * target_1_stds) + target_1_means
   targets_2 = (targets_2 * target_2_stds) + target_2_means

   predictions_1 = pd.DataFrame(predictions[0], columns = targets_1.columns)
   predictions_2 = pd.DataFrame(predictions[1], columns = targets_2.columns)   
   predictions_1 = (predictions_1 * target_1_stds) + target_1_means  
   predictions_2 = (predictions_2 * target_2_stds) + target_2_means     
   
   inputs = inputs[mask]
   targets_1 = targets_1[mask]
   targets_2 = targets_2[mask]
   targets = pd.concat([targets_1,targets_2], axis = 1)
   predictions = pd.concat([predictions_1,predictions_2], axis = 1)
   predictions = predictions[mask]
   cpweights.reset_index(drop=True,inplace=True)
   cpweights = cpweights[mask]
   cat = cat[mask]

   inputs.reset_index(drop=True,inplace=True)
   targets.reset_index(drop=True,inplace=True)
   predictions.reset_index(drop=True,inplace=True)
   print(len(targets),len(predictions))
   cpweights.reset_index(drop=True,inplace=True)
   cat.reset_index(drop=True,inplace=True)
  
   nevents = len(inputs)
   print("Total Number of events after DM cut:", nevents)
      
   ip1 = [ROOT.TLorentzVector(inputs['ip_x_1'].values[index],inputs['ip_y_1'].values[index],inputs['ip_z_1'].values[index],0.) for index in range(nevents)]
   ip2 = [ROOT.TLorentzVector(inputs['ip_x_2'].values[index],inputs['ip_y_2'].values[index],inputs['ip_z_2'].values[index],0.) for index in range(nevents)]

   nu1_true = [ROOT.TLorentzVector(targets['gen_nu_px_1'].values[index],targets['gen_nu_py_1'].values[index],targets['gen_nu_pz_1'].values[index],math.sqrt(targets['gen_nu_px_1'].values[index]**2+targets['gen_nu_py_1'].values[index]**2+targets['gen_nu_pz_1'].values[index]**2)) for index in range(nevents)]
   nu2_true = [ROOT.TLorentzVector(targets['gen_nu_px_2'].values[index],targets['gen_nu_py_2'].values[index],targets['gen_nu_pz_2'].values[index],math.sqrt(targets['gen_nu_px_2'].values[index]**2+targets['gen_nu_py_2'].values[index]**2+targets['gen_nu_pz_2'].values[index]**2)) for index in range(nevents)]
   nu1_pred = [ROOT.TLorentzVector(predictions['gen_nu_px_1'].values[index],predictions['gen_nu_py_1'].values[index],predictions['gen_nu_pz_1'].values[index],math.sqrt(predictions['gen_nu_px_1'].values[index]**2+predictions['gen_nu_py_1'].values[index]**2+predictions['gen_nu_pz_1'].values[index]**2)) for index in range(nevents)]
   nu2_pred = [ROOT.TLorentzVector(predictions['gen_nu_px_2'].values[index],predictions['gen_nu_py_2'].values[index],predictions['gen_nu_pz_2'].values[index],math.sqrt(predictions['gen_nu_px_2'].values[index]**2+predictions['gen_nu_py_2'].values[index]**2+predictions['gen_nu_pz_2'].values[index]**2)) for index in range(nevents)]
         
   pi_1 = [Particle(ROOT.TLorentzVector(inputs['pi_px_1'].values[index],inputs['pi_py_1'].values[index],inputs['pi_pz_1'].values[index],inputs['pi_E_1'].values[index]),0.) for index in range(nevents)]
   pi_2 = [Particle(ROOT.TLorentzVector(inputs['pi_px_2'].values[index],inputs['pi_py_2'].values[index],inputs['pi_pz_2'].values[index],inputs['pi_E_2'].values[index]),0.) for index in range(nevents)]
      
   pi2_1 = [Particle(ROOT.TLorentzVector(inputs['pi2_px_1'].values[index],inputs['pi2_py_1'].values[index],inputs['pi2_pz_1'].values[index],inputs['pi2_E_1'].values[index]),0.) for index in range(nevents)]
   pi2_2 = [Particle(ROOT.TLorentzVector(inputs['pi2_px_2'].values[index],inputs['pi2_py_2'].values[index],inputs['pi2_pz_2'].values[index],inputs['pi2_E_2'].values[index]),0.) for index in range(nevents)]   

   pi3_1 = [Particle(ROOT.TLorentzVector(inputs['pi3_px_1'].values[index],inputs['pi3_py_1'].values[index],inputs['pi3_pz_1'].values[index],inputs['pi3_E_1'].values[index]),0.) for index in range(nevents)]
   pi3_2 = [Particle(ROOT.TLorentzVector(inputs['pi3_px_2'].values[index],inputs['pi3_py_2'].values[index],inputs['pi3_pz_2'].values[index],inputs['pi3_E_2'].values[index]),0.) for index in range(nevents)]  

   pi0_1 = [Particle(ROOT.TLorentzVector(inputs['pi0_px_1'].values[index],inputs['pi0_py_1'].values[index],inputs['pi0_pz_1'].values[index],inputs['pi0_E_1'].values[index]), 0.) for index in range(nevents)]
   pi0_2 = [Particle(ROOT.TLorentzVector(inputs['pi0_px_2'].values[index],inputs['pi0_py_2'].values[index],inputs['pi0_pz_2'].values[index],inputs['pi0_E_2'].values[index]), 0.) for index in range(nevents)]

   vis1=[pi_1[index].Fourvector + pi2_1[index].Fourvector + pi3_1[index].Fourvector +  pi0_1[index].Fourvector for index in range(nevents)]
   vis2=[pi_2[index].Fourvector + pi2_2[index].Fourvector + pi3_2[index].Fourvector +  pi0_2[index].Fourvector for index in range(nevents)]

   tau1_true = [Particle(vis1[index] + nu1_true[index],0.) for index in range(nevents)]
   tau2_true = [Particle(vis2[index] + nu2_true[index],0.) for index in range(nevents)] 
   tau1_pred = [Particle(vis1[index] + nu1_pred[index],0.) for index in range(nevents)]
   tau2_pred = [Particle(vis2[index] + nu2_pred[index],0.) for index in range(nevents)]   

   PV1_true = []
   PV2_true = []
   PV1_pred = []
   PV2_pred = []
   Aco = []
   Aco_Mix = []
   
   for index in range(nevents):
      if (index % 10000 == 0): print(index)
      if cat["tau_decay_mode_1"].values[index] == 0 and cat["tau_decay_mode_2"].values[index] == 0:
         PV1_true.append(PolarimetricVector([pi_1[index]], [], tau1_true[index]))
         PV2_true.append(PolarimetricVector([pi_2[index]], [], tau2_true[index]))
         PV1_pred.append(PolarimetricVector([pi_1[index]], [], tau1_pred[index]))
         PV2_pred.append(PolarimetricVector([pi_2[index]], [], tau2_pred[index]))       
         Aco.append(AcoplanarityAngle(ip1[index],ip2[index],pi_1[index].Fourvector,pi_2[index].Fourvector,type1="IP",type2="IP"))
         Aco_Mix.append(AcoplanarityAngle(PV1_pred[index],PV2_pred[index],tau1_pred[index].Fourvector,tau2_pred[index].Fourvector,type1="PV",type2="PV"))

   x = pd.DataFrame()
   y = pd.DataFrame()      
   x['PV_Aco'] = [AcoplanarityAngle_PV(PV1_true[index],PV2_true[index],tau1_true[index].Fourvector,tau2_true[index].Fourvector) for index in range(nevents)]
   y['PV_Aco'] = [AcoplanarityAngle_PV(PV1_pred[index],PV2_pred[index],tau1_pred[index].Fourvector,tau2_pred[index].Fourvector) for index in range(nevents)]
   y['Aco'] = Aco
   y['Aco_Mix'] = Aco_Mix 

   PV1_true = [ROOT.TLorentzVector(targets['PV1_x'].values[index],targets['PV1_y'].values[index],targets['PV1_z'].values[index],targets['PV1_E'].values[index]) for index in range(nevents)]
   PV2_true = [ROOT.TLorentzVector(targets['PV2_x'].values[index],targets['PV2_y'].values[index],targets['PV2_z'].values[index],targets['PV2_E'].values[index]) for index in range(nevents)] 
   PV1_pred = [ROOT.TLorentzVector(predictions['PV1_x'].values[index],predictions['PV1_y'].values[index],predictions['PV1_z'].values[index],predictions['PV1_E'].values[index]) for index in range(nevents)]
   PV2_pred = [ROOT.TLorentzVector(predictions['PV2_x'].values[index],predictions['PV2_y'].values[index],predictions['PV2_z'].values[index],predictions['PV2_E'].values[index]) for index in range(nevents)]   

   x['PV_Aco_reg'] = [AcoplanarityAngle_PV(PV1_true[index],PV2_true[index],tau1_true[index].Fourvector,tau2_true[index].Fourvector) for index in range(nevents)]
   y['PV_Aco_reg'] = [AcoplanarityAngle_PV(PV1_pred[index],PV2_pred[index],tau1_pred[index].Fourvector,tau2_pred[index].Fourvector) for index in range(nevents)]


   savedir = path + "DM" + DM
   if not os.path.exists(savedir):
      os.makedirs(savedir)

   bins = 30
   r = (0, 2 * np.pi)

   if DM == "00":
      AcoPlot(bins,r,x['PV_Aco'].values,cpweights,"Acolanarity angle distribution for $DM_{1}$ = 0 & $DM_{2}$ = 0 PV+PV",savedir+"/PV_Aco_True.png".format(DM))
      AcoPlot(bins,r,y['PV_Aco'].values,cpweights,"Acolanarity angle distribution for $DM_{1}$ = 0 & $DM_{2}$ = 0 PV+PV",savedir+"/PV_Aco_Pred.png".format(DM))
      AcoPlot(bins,r,x['PV_Aco_reg'].values,cpweights,"Acolanarity angle distribution for $DM_{1}$ = 0 & $DM_{2}$ = 0 $PV_{reg}$+$PV_{reg}$",savedir+"/PV_Aco_True_reg.png".format(DM))
      AcoPlot(bins,r,y['PV_Aco_reg'].values,cpweights,"Acolanarity angle distribution for $DM_{1}$ = 0 & $DM_{2}$ = 0 $PV_{reg}$+$PV_{reg}$",savedir+"/PV_Aco_Pred_reg.png".format(DM))   
      AcoPlot(bins,r,y['Aco'].values,cpweights,"Acolanarity angle distribution for $DM_{1}$ = 0 & $DM_{2}$ = 0 (CP Analysis)",savedir+"/Aco.png".format(DM))
      AcoPlot(bins,r,y['Aco_Mix'].values,cpweights,"Acolanarity angle distribution for $DM_{1}$ = 0 & $DM_{2}$ = 0 Mix",savedir+"/Aco_Mix.png".format(DM))

   plots = {
      'gen_nu_px_1': ["True","Regression",r'$\nu_1$ $P_x$',"Entries",-100,100,40],
      'gen_nu_py_1': ["True","Regression",r'$\nu_1$ $P_y$',"Entries",-100,100,40],
      'gen_nu_pz_1': ["True","Regression",r'$\nu_1$ $P_z$',"Entries",-100,100,40],
      'gen_nu_px_2': ["True","Regression",r'$\nu_2$ $P_x$',"Entries",-100,100,40],
      'gen_nu_py_2': ["True","Regression",r'$\nu_2$ $P_y$',"Entries",-100,100,40],
      'gen_nu_pz_2': ["True","Regression",r'$\nu_2$ $P_z$',"Entries",-100,100,40],
      'PV1_x': ["True","Regression",r'$PV_1$ $P_x$',"Entries",-100,100,40],
      'PV1_y': ["True","Regression",r'$PV_1$ $P_y$',"Entries",-100,100,40],
      'PV1_z': ["True","Regression",r'$PV_1$ $P_z$',"Entries",-100,100,40],
      'PV1_E': ["True","Regression",r'$PV_1$ $E$',"Entries",-100,100,40],
      'PV2_x': ["True","Regression",r'$PV_2$ $P_x$',"Entries",-100,100,40],
      'PV2_y': ["True","Regression",r'$PV_2$ $P_y$',"Entries",-100,100,40],
      'PV2_z': ["True","Regression",r'$PV_2$ $P_z$',"Entries",-100,100,40],
      'PV2_E': ["True","Regression",r'$PV_2$ $E$',"Entries",-100,100,40],
   }

   for key in plots:
      PlotHistograms(targets[key],predictions[key],plots[key][0],plots[key][1],plots[key][2],plots[key][3],plots[key][4],plots[key][5],plots[key][6], savedir + "/" + key)
   
if __name__ == "__main__":   
   parser = argparse.ArgumentParser(description="Load .pkl file and process it")
   parser.add_argument("-DM", "--DM", type=str, help="Decay Modes to process")
   parser.add_argument("-p","--path",type=str, help="Path to input directory")
   args = parser.parse_args()
   main(args)
