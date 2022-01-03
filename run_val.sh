#!/bin/bash

rm -f ~/slurmout/*
rm -rf /hpc/group/laberlabs/jml165/valresults/
mkdir /hpc/group/laberlabs/jml165/valresults
rm -f /hpc/group/laberlabs/jml165/valcombresults/*.csv

sbatch ~/rl/lawson-bandits/val.sh
#wait

#module load R
#Rscipt ~/combine.R
