#!/bin/bash

rm -f ~/slurmout/*
rm -rf ~/rl/valneuralnets/
mkdir ~/rl/valneuralnets
#rm -f ~/rl/valcombresults/*.csv

sbatch fitnn.sh
#wait

#module load R
#Rscipt ~/combine.R
