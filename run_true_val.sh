#!/bin/bash

rm -f ~/slurmout/*
rm -rf /hpc/group/laberlabs/jml165/truevalresults/
mkdir /hpc/group/laberlabs/jml165/truevalresults
rm -f /hpc/group/laberlabs/jml165/truevalcombresults/*.csv

sbatch ~/rl/lawson-bandits/true_val.sh
#wait

#module load R
#Rscipt ~/combine.R
