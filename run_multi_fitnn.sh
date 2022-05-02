#!/bin/bash

rm -f ~/slurmout/*
rm -rf ~/rl/multivalneuralnets/
mkdir ~/rl/multivalneuralnets
#rm -f ~/rl/valcombresults/*.csv

sbatch ~/rl/lawson-bandits/multi_fitnn.sh
#wait

#module load R
#Rscipt ~/combine.R
