#!/bin/bash

rm -f ~/slurmout/*
rm -rf ~/rl/valresults/
mkdir ~/rl/valresults 
rm -f ~/rl/valcombresults/*.csv

sbatch val.sh
#wait

#module load R
#Rscipt ~/combine.R
