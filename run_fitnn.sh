#!/bin/bash

rm -f ~/slurmout/*
rm -rf ~/rl/valneuralnets/
mkdir ~/rl/valneuralnets
#rm -f ~/rl/valcombresults/*.csv

sbatch ~/rl/lawson-bandits/fitnn.sh
#wait

#module load R
#Rscipt ~/combine.R
