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

export CUDA_VISIBLE_DEVICES=0
echo "[INFO] Using GPU $CUDA_VISIBLE_DEVICES"

pip install -e .
# pip install -U kaleido

### Run the eval_blimp.sh script
for folder in newest_5 newest_10 newest_15 newest_20 newest_25 newest_30 newest_40 newest_50; do
    mkdir -p ./$folder
    python ./llama_content_fixed.py $folder
    python ./clean_customname.py $folder
    python ./cossine.py $folder
done


echo "----------- Job $SLURM_JOBID complete"

