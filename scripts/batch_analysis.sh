#!/bin/bash
#$ -q gpu.q
#$ -l h_rt=24:0:0
#$ -t 1-4

source /vols/cms/ks1021/offline/Htt_CPinDecay/TauMLTools/env.sh conda
cd /vols/cms/ks1021/offline/Htt_CPinDecay/Regression

# List of options
options=("00" "10" "01")

# Get the option corresponding to the task ID
opt=${options[$SGE_TASK_ID-1]}

# Submit the job with the selected option
python Neutrino/analysis.py -p "Neutrino/store/T2/" -DM "$opt"
echo "Job submitted with option -DM $opt
