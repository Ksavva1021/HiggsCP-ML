#!/bin/sh
#$ -q gpu.q
#$ -l h_rt=24:0:0
#$ -m bea
#$ -M ks1021@ic.ac.uk

source /vols/cms/ks1021/offline/Htt_CPinDecay/TauMLTools/env.sh conda
cd /vols/cms/ks1021/offline/Htt_CPinDecay/Regression

log_file="samples/logs/inputs_ggH.log"  # Replace with the actual path and filename you want to use

python python/PrepareData.py -d "samples/trees/" -o "samples/pickle/ggH.pkl" > "$log_file" 2>&1
echo "Script successfully submitted to batch"

