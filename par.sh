#!/bin/bash
#SBATCH -J jlcb1
#SBATCH --array 1-500
#SBATCH --cpus-per-task 1
#SBATCH --output="/hpc/home/jml165/slurmout/R-%A_%a.out"
#SBATCH --error="/hpc/home/jml165/slurmout/R-%A_%a.err"
#SBATCH --mem=2000M
#SBATCH --partition=scavenger
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jml165@duke.edu
#SBATCH --exclude=dcc-ultrasound-04,dcc-ultrasound-06,dcc-ultrasound-07

module purge
module load Julia/1.6.1

julia -t 1 ~/rl/lawson-bandits/main.jl
