#!/bin/bash

#SBATCH --job-name=content-semisupervised-modified
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mail-type=end
#SBATCH --mail-user=hnguye30@villanova.edu

#### Load necessary modules
module load python/3.8

export CUDA_VISIBLE_DEVICES=0
echo "[INFO] Using GPU $CUDA_VISIBLE_DEVICES"


echo "---------- Start job $SLURM_JOBID"

source /home/hnguye30/miniconda3/etc/profile.d/conda.sh
conda activate topic

pip install -e .

#### Run the eval_blimp.sh script
python ./semisup_content_modified.py

echo "----------- Job $SLURM_JOBID complete"

