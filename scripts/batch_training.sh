#!/bin/sh
#$ -q gpu.q
#$ -l h_rt=24:0:0
#$ -m bea
#$ -M ks1021@ic.ac.uk

cd /vols/cms/ks1021/offline/Htt_CPinDecay/Regression
source /vols/cms/ks1021/offline/Htt_CPinDecay/TauMLTools/env.sh conda

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 -t <value_t> -s <value_s> -l <log_file_name> [additional_args]"
  exit 1
fi

t=""
s=""
log_file=""

while [ $# -gt 0 ]; do
    case "$1" in
        -t)
            t="$2"
            shift 2
            ;;
        -s)
            s="$2"
            shift 2
            ;;
        -l)
            log_file="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

mkdir -p "$s"

python Training/train.py -t "$t" -s "$s" "$@" > "$log_file" 2>&1
echo "Script successfully submitted to batch"
