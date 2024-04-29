#!/bin/bash -l

#$ -P ace-ig          # SCC project name
#$ -l h_rt=06:00:00   # Specify the hard time limit for the job
#$ -N bce+dice      # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -l gpus=1 
#$ -l gpu_c=7


module load python3/3.8
source /projectnb/ec500kb/projects/UniverSeg/univer_seg_venv/bin/activate

python /projectnb/ec500kb/projects/UniverSeg/code/tmp/main_different_loss.py
       --experiment_name="bce+dice"
