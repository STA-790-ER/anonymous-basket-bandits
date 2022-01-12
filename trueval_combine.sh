#!/bin/bash
#SBATCH -J jlcb1
#SBATCH --array 1-100
#SBATCH --cpus-per-task 1
#SBATCH --output="/hpc/home/jml165/slurmout/R-%A_%a.out"
#SBATCH --error="/hpc/home/jml165/slurmout/R-%A_%a.err"
#SBATCH --mem=32G
#SBATCH --partition=common,scavenger
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jml165@duke.edu
#SBATCH --exclude=dcc-ultrasound-04,dcc-ultrasound-06,dcc-ultrasound-07

module purge

module load R

Rscript ~/rl/lawson-bandits/trueval_combine.R 
