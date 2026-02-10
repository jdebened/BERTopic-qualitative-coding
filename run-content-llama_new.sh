#!/bin/bash

#SBATCH --job-name=content-llama-new
#SBATCH --time=24:00:00
#SBATCH --account=comp_sci
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mail-type=end
#SBATCH --mail-user=hnguye30@villanova.edu

#### Load necessary modules
module load python/3.8

echo "---------- Start job $SLURM_JOBID"

source /home/hnguye30/miniconda3/etc/profile.d/conda.sh
conda activate topic_new

pip install -e .
# pip install -U kaleido

#### Run the eval_blimp.sh script
python ./llama_content_new.py
python ./clean_customname.py new
python ./cossine.py new

echo "----------- Job $SLURM_JOBID complete"

