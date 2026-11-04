#!/bin/bash

#SBATCH --job-name=content-llama-new
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
for folder in newest_5_outlier newest_10_outlier newest_15_outlier newest_20_outlier newest_25_outlier newest_30_outlier newest_40_outlier newest_50_outlier; do
    mkdir -p ./$folder
    python ./llama_content_fixed_outlier.py $folder
    python ./clean_customname.py $folder
    python ./cossine.py $folder
done


echo "----------- Job $SLURM_JOBID complete"

