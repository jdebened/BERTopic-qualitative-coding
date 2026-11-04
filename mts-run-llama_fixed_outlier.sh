#!/bin/bash

#SBATCH --job-name=mts-llama-fixed
#SBATCH --time=40:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mail-type=end
#SBATCH --mail-user=hnguye30@villanova.edu

#### Load necessary modules
module load python/3.8

echo "---------- Start job $SLURM_JOBID"

source /home/hnguye30/miniconda3/etc/profile.d/conda.sh
conda activate topic_new

export CUDA_VISIBLE_DEVICES=1
echo "[INFO] Using GPU $CUDA_VISIBLE_DEVICES"

pip install -e .
# pip install -U kaleido

### Run the eval_blimp.sh script
for folder in outlier_5 outlier_10 outlier_15 outlier_20 outlier_25; do
    mkdir -p ./mts/$folder
    python ./mts_llama_fixed_outlier.py ./mts/$folder
    python ./mts_clean_customname.py ./mts/$folder
    python ./mts_cossine.py ./mts/$folder
done


echo "----------- Job $SLURM_JOBID complete"

