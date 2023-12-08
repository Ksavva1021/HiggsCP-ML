import os
import sys
directory = os.getcwd()
sys.path.append(directory+'/python/')
from Particle import Particle
from Utilities import PolarimetricVector, AcoplanarityAngle_PV, AcoplanarityAngle

import uproot3
import pandas as pd
import argparse
import numpy as np
import ROOT
import math

pd.set_option('display.max_columns', None)
pd.set_option("display.precision", 2)

def main(filename):
   file = uproot3.open(filename)
   
   # Define the list of variables to be extracted from the ROOT file
   variables = [
   'pi_px_1', 'pi_py_1', 'pi_pz_1', 'pi_E_1', 
   'pi_px_2', 'pi_py_2', 'pi_pz_2', 'pi_E_2',
   'pi2_px_1', 'pi2_py_1', 'pi2_pz_1', 'pi2_E_1',
   'pi2_px_2', 'pi2_py_2', 'pi2_pz_2', 'pi2_E_2',
   'pi3_px_1', 'pi3_py_1', 'pi3_pz_1', 'pi3_E_1', 
   'pi3_px_2', 'pi3_py_2', 'pi3_pz_2', 'pi3_E_2',
   'pi0_px_1', 'pi0_py_1', 'pi0_pz_1', 'pi0_E_1',
   'pi0_px_2', 'pi0_py_2', 'pi0_pz_2', 'pi0_E_2',
   'ip_x_1', 'ip_y_1', 'ip_z_1','ipcov00_1','ipcov01_1','ipcov02_1','ipcov10_1','ipcov11_1','ipcov12_1','ipcov20_1','ipcov21_1','ipcov22_1',
   'ip_x_2', 'ip_y_2', 'ip_z_2','ipcov00_2','ipcov01_2','ipcov02_2','ipcov10_2','ipcov11_2','ipcov12_2','ipcov20_2','ipcov21_2','ipcov22_2',
   'ip_sig_1','ip_sig_2',
   'sv_x_1', 'sv_y_1', 'sv_z_1','svcov00_1','svcov01_1','svcov02_1','svcov10_1','svcov11_1','svcov12_1','svcov20_1','svcov21_1','svcov22_1',
   'sv_x_2', 'sv_y_2', 'sv_z_2','svcov00_2','svcov01_2','svcov02_2','svcov10_2','svcov11_2','svcov12_2','svcov20_2','svcov21_2','svcov22_2',
   'pt_1', 'pt_2', 'pt_vis',
   'met', 'metx', 'mety',
   'gen_tau_metx','gen_tau_mety','gen_metx','gen_mety',
   'tau_decay_mode_1','tau_decay_mode_2',
   'q1', 'q2',
   'q_pi_1','q_pi2_1','q_pi3_1',
   'q_pi_2','q_pi2_2','q_pi3_2', 
   'wt_cp_sm', 'wt_cp_ps', 'wt_cp_mm',
   'deepTauVsJets_medium_1', 'deepTauVsJets_medium_2', 'os',
   'gen_nu_px_1','gen_nu_py_1','gen_nu_pz_1',
   'gen_nu_px_2','gen_nu_py_2','gen_nu_pz_2',
   'gen_vis_px_1','gen_vis_py_1','gen_vis_pz_1','gen_vis_E_1',
   'gen_vis_px_2','gen_vis_py_2','gen_vis_pz_2','gen_vis_E_2',
   'gen_neutral_px_1','gen_neutral_py_1','gen_neutral_pz_1','gen_neutral_E_1',
   'gen_neutral_px_2','gen_neutral_py_2','gen_neutral_pz_2','gen_neutral_E_2'
   ]
   
   print("Processing:",filename)
   # Convert ROOT tree to pandas DataFrame
   df = file['ntuple'].pandas.df(variables)
   df_pions = df.iloc[:,0:32]
   df = df.loc[(df_pions != 0).any(axis=1)]
  
   # Apply selections to filter events
   df = df[(df['tau_decay_mode_1'] == 0) | (df['tau_decay_mode_1'] == 1)]
   df = df[(df['tau_decay_mode_2'] == 0) | (df['tau_decay_mode_2'] == 1)]
   df = df[(df['deepTauVsJets_medium_1'] > 0.5) & (df['deepTauVsJets_medium_2'] > 0.5)]
   df = df[(df['ip_sig_1'] > 1.0) & (df['ip_sig_2'] > 1.0)]
   df = df[(df['os'] == 1)]
   df = df[~(df < -9990).any(axis=1)]
   df.reset_index(drop=True,inplace = True)
   nevents = len(df)  
   
   # Flip sign of selected columns based on the charge of particles
   variables_to_flip = [
   'pi_px','pi_py','pi_pz','pi_E',
   'pi2_px','pi2_py','pi2_pz','pi2_E',
   'pi3_px','pi3_py','pi3_pz','pi3_E',
   'pi0_px','pi0_py','pi0_pz','pi0_E',
   'ip_x','ip_y','ip_z','ipcov00','ipcov01','ipcov02','ipcov10','ipcov11','ipcov12','ipcov20','ipcov21','ipcov22',
   'ip_sig',
   'sv_x','sv_y','sv_z','svcov00','svcov01','svcov02','svcov10','svcov11','svcov12','svcov20','svcov21','svcov22',
   'pt','tau_decay_mode',   
   'q_pi','q_pi2','q_pi3',   
   'gen_nu_px','gen_nu_py','gen_nu_pz',
   'gen_vis_px','gen_vis_py','gen_vis_pz','gen_vis_E',
   'gen_neutral_px','gen_neutral_py','gen_neutral_pz','gen_neutral_E'
   ]  
   mask=df['q1']<df['q2'] 
   for col in variables_to_flip:
      if col+'_1' in df.columns and col+'_2' in df.columns:
         df.loc[mask, col + '_1'], df.loc[mask, col + '_2'] = df.loc[mask, col + '_2'], df.loc[mask, col + '_1']

   # Drop NaNs
   df = df.dropna(how='any')
   df.reset_index(drop=True,inplace=True)

   nevents = len(df)
   
   # Calculate Polarimetric Vectors
   ip1 = [ROOT.TLorentzVector(df['ip_x_1'].values[index],df['ip_y_1'].values[index],df['ip_z_1'].values[index],0.) for index in range(nevents)]
   ip2 = [ROOT.TLorentzVector(df['ip_x_2'].values[index],df['ip_y_2'].values[index],df['ip_z_2'].values[index],0.) for index in range(nevents)]
   nu1_true = [ROOT.TLorentzVector(df['gen_nu_px_1'].values[index],df['gen_nu_py_1'].values[index],df['gen_nu_pz_1'].values[index],math.sqrt(df['gen_nu_px_1'].values[index]**2+df['gen_nu_py_1'].values[index]**2+df['gen_nu_pz_1'].values[index]**2)) for index in range(nevents)]
   nu2_true = [ROOT.TLorentzVector(df['gen_nu_px_2'].values[index],df['gen_nu_py_2'].values[index],df['gen_nu_pz_2'].values[index],math.sqrt(df['gen_nu_px_2'].values[index]**2+df['gen_nu_py_2'].values[index]**2+df['gen_nu_pz_2'].values[index]**2)) for index in range(nevents)]
   pi_1 = [Particle(ROOT.TLorentzVector(df['pi_px_1'].values[index],df['pi_py_1'].values[index],df['pi_pz_1'].values[index],df['pi_E_1'].values[index]),0.) for index in range(nevents)]
   pi_2 = [Particle(ROOT.TLorentzVector(df['pi_px_2'].values[index],df['pi_py_2'].values[index],df['pi_pz_2'].values[index],df['pi_E_2'].values[index]),0.) for index in range(nevents)]
   pi2_1 = [Particle(ROOT.TLorentzVector(df['pi2_px_1'].values[index],df['pi2_py_1'].values[index],df['pi2_pz_1'].values[index],df['pi2_E_1'].values[index]),0.) for index in range(nevents)]
   pi2_2 = [Particle(ROOT.TLorentzVector(df['pi2_px_2'].values[index],df['pi2_py_2'].values[index],df['pi2_pz_2'].values[index],df['pi2_E_2'].values[index]),0.) for index in range(nevents)]   
   pi3_1 = [Particle(ROOT.TLorentzVector(df['pi3_px_1'].values[index],df['pi3_py_1'].values[index],df['pi3_pz_1'].values[index],df['pi3_E_1'].values[index]),0.) for index in range(nevents)]
   pi3_2 = [Particle(ROOT.TLorentzVector(df['pi3_px_2'].values[index],df['pi3_py_2'].values[index],df['pi3_pz_2'].values[index],df['pi3_E_2'].values[index]),0.) for index in range(nevents)]  
   pi0_1 = [Particle(ROOT.TLorentzVector(df['pi0_px_1'].values[index],df['pi0_py_1'].values[index],df['pi0_pz_1'].values[index],df['pi0_E_1'].values[index]), 0.) for index in range(nevents)]
   pi0_2 = [Particle(ROOT.TLorentzVector(df['pi0_px_2'].values[index],df['pi0_py_2'].values[index],df['pi0_pz_2'].values[index],df['pi0_E_2'].values[index]), 0.) for index in range(nevents)]
   vis1=[pi_1[index].Fourvector + pi2_1[index].Fourvector + pi3_1[index].Fourvector +  pi0_1[index].Fourvector for index in range(nevents)]
   vis2=[pi_2[index].Fourvector + pi2_2[index].Fourvector + pi3_2[index].Fourvector +  pi0_2[index].Fourvector for index in range(nevents)]
   tau1_true = [Particle(vis1[index] + nu1_true[index],0.) for index in range(nevents)]
   tau2_true = [Particle(vis2[index] + nu2_true[index],0.) for index in range(nevents)]
   
   PV1_true = []
   PV2_true = []
   
   for index in range(nevents):
      if df['tau_decay_mode_1'].values[index] == 0 and df['tau_decay_mode_2'].values[index] == 0:  
         PV1_true.append(PolarimetricVector([pi_1[index]], [], tau1_true[index]))
         PV2_true.append(PolarimetricVector([pi_2[index]], [], tau2_true[index]))  
      if df['tau_decay_mode_1'].values[index] == 0 and df['tau_decay_mode_2'].values[index] == 1:      
         PV1_true.append(PolarimetricVector([pi_1[index]], [], tau1_true[index]))
         PV2_true.append(PolarimetricVector([pi_2[index]], [pi0_2[index]], tau2_true[index])) 
      if df['tau_decay_mode_1'].values[index] == 1 and df['tau_decay_mode_2'].values[index] == 0:      
         PV1_true.append(PolarimetricVector([pi_1[index]], [pi0_1[index]], tau1_true[index]))
         PV2_true.append(PolarimetricVector([pi_2[index]], [], tau2_true[index]))
      if df['tau_decay_mode_1'].values[index] == 1 and df['tau_decay_mode_2'].values[index] == 1:      
         PV1_true.append(PolarimetricVector([pi_1[index]], [pi0_1[index]], tau1_true[index]))
         PV2_true.append(PolarimetricVector([pi_2[index]], [pi0_2[index]], tau2_true[index]))

   # Add PolarimetricVector-related columns to the DataFrame
   df['PV_Aco'] = [AcoplanarityAngle_PV(PV1_true[index],PV2_true[index],tau1_true[index].Fourvector,tau2_true[index].Fourvector) for index in range(nevents)]
   df['PV1_x'] = [PV1_true[index].Px() for index in range(nevents)]   
   df['PV1_y'] = [PV1_true[index].Py() for index in range(nevents)]   
   df['PV1_z'] = [PV1_true[index].Pz() for index in range(nevents)]   
   df['PV1_E'] = [PV1_true[index].E() for index in range(nevents)]
   df['PV2_x'] = [PV2_true[index].Px() for index in range(nevents)]   
   df['PV2_y'] = [PV2_true[index].Py() for index in range(nevents)]   
   df['PV2_z'] = [PV2_true[index].Pz() for index in range(nevents)] 
   df['PV2_E'] = [PV2_true[index].E() for index in range(nevents)]

   # Remove rows with NaN values
   df = df.dropna(how='any')
   df.reset_index(drop=True,inplace=True)

   print("Finished Processing:", filename)
   return df 

if __name__ == "__main__":
   # Parse command-line arguments
   parser = argparse.ArgumentParser(description="Load variables from a root file and save them to a .pkl file.")
   parser.add_argument("-d", "--input_dir", type=str, help="The directory containing input root files")
   parser.add_argument("-o","--output_file",type=str, help="The name of the output file")
   args = parser.parse_args()
   
   # Create a list of input files
   input_files = [os.path.join(args.input_dir, filename) for filename in os.listdir(args.input_dir) if filename.endswith(".root") and not filename.startswith('VBF')]   
   # Process each input file and store the results in a dictionary
   df = {str(index): main(filename) for index,filename in enumerate(input_files)}
   # Concatenate the DataFrames from different files into one   
   merged_df = pd.concat(list(df.values()), axis=0)
   merged_df.reset_index(drop=True, inplace=True)
   merged_df['event_number'] = merged_df.index
   
   merged_df.to_pickle(args.output_file)
   
   # Count the number of NaN values in the merged DataFrame
   nan_count = merged_df.isna().sum().sum()
   print("Finished\nList of Columns\n", merged_df.columns.to_list(),"\nNumber of NaNs:",nan_count)

