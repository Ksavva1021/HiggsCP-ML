import os
import uproot3
import pandas as pd
import argparse
import numpy as np
import ROOT

pd.set_option('display.max_columns', None)
pd.set_option("display.precision", 2)

def main(filename):
   file = uproot3.open(filename)
   variables = [
   'pi_px_1', 'pi_py_1', 'pi_pz_1', 'pi_E_1', 
   'pi_px_2', 'pi_py_2', 'pi_pz_2', 'pi_E_2',
   'pi2_px_1', 'pi2_py_1', 'pi2_pz_1', 'pi2_E_1',
   'pi2_px_2', 'pi2_py_2', 'pi2_pz_2', 'pi2_E_2',
   'pi3_px_1', 'pi3_py_1', 'pi3_pz_1', 'pi3_E_1', 
   'pi3_px_2', 'pi3_py_2', 'pi3_pz_2', 'pi3_E_2',
   'pi0_px_1', 'pi0_py_1', 'pi0_pz_1', 'pi0_E_1',
   'pi0_px_2', 'pi0_py_2', 'pi0_pz_2', 'pi0_E_2',
   'dm_1','dm_2',
   'tau1_charge', 'tau2_charge',
   'wt_cp_sm', 'wt_cp_ps', 'wt_cp_mm',
   'nu_px_1','nu_py_1','nu_pz_1',
   'nu_px_2','nu_py_2','nu_pz_2',
   'pv_aco_angle'
   ]
   print("Processing:",filename)
   df = file['ntuple'].pandas.df(variables)
   df_pions = df.iloc[:,0:32]
   df = df.loc[(df_pions != 0).any(axis=1)]
  
   # Applying selections
   df = df[(df['dm_1'] == 0) | (df['dm_1'] == 1) | (df['dm_1'] == 10) | (df['dm_1'] == 11)]
   df = df[(df['dm_2'] == 0) | (df['dm_2'] == 1) | (df['dm_2'] == 10) | (df['dm_2'] == 11)]
   df = df[~(df < -9990).any(axis=1)]
   df.reset_index(drop=True,inplace = True)
   nevents = len(df)  
   
   # q1 = positive / q2 = negative
   variables_to_flip = [
   'pi_px','pi_py','pi_pz','pi_E',
   'pi2_px','pi2_py','pi2_pz','pi2_E',
   'pi3_px','pi3_py','pi3_pz','pi3_E',
   'pi0_px','pi0_py','pi0_pz','pi0_E',
   'dm',   
   'nu_px','nu_py','nu_pz',
   ]
   
   mask=df['tau1_charge']<df['tau2_charge'] 
   for col in variables_to_flip:
      if col+'_1' in df.columns and col+'_2' in df.columns:
         df.loc[mask, col + '_1'], df.loc[mask, col + '_2'] = df.loc[mask, col + '_2'], df.loc[mask, col + '_1']

   df = df.dropna(how='any')
   df.reset_index(drop=True,inplace=True)
   print("Finished Processing:", filename)
   return df 

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Load variables from a root file and save them to a .pkl file.")
   parser.add_argument("-d", "--input_dir", type=str, help="The directory containing input root files")
   parser.add_argument("-o","--output_file",type=str, help="The name of the output file")
   args = parser.parse_args()
   input_files = [os.path.join(args.input_dir, filename) for filename in os.listdir(args.input_dir) if filename.endswith(".root") and not filename.startswith('VBF') and '2018' in filename]   
   df = {str(index): main(filename) for index,filename in enumerate(input_files)}
   merged_df = pd.concat(list(df.values()), axis=0)
   merged_df.reset_index(drop=True, inplace=True)
   merged_df['event_number'] = merged_df.index
   merged_df.to_pickle(args.output_file)
   nan_count = merged_df.isna().sum().sum()
   print("Finished\nList of Columns\n", merged_df.columns.to_list(),"\nNumber of NaNs:",nan_count)

