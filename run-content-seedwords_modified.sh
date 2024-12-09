#!/bin/bash

#SBATCH --job-name=content-seedwords-modified-0
#SBATCH --time=06:00:00
#SBATCH --account=comp_sci
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mail-type=end
#SBATCH --mail-user=hnguye30@villanova.edu

#### Load necessary modules
module load python/3.8

echo "---------- Start job $SLURM_JOBID"

source /home/hnguye30/miniconda3/etc/profile.d/conda.sh
conda activate topic

pip install -e .

#### Run the eval_blimp.sh script
python ./seedwords_content_modified.py 0 politics celebrity finance music gender retail food drug suicide brain

echo "----------- Job $SLURM_JOBID complete"

