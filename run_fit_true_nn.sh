#!/bin/bash

rm -f ~/slurmout/*
rm -rf ~/rl/truevalneuralnets/
mkdir ~/rl/truevalneuralnets
#rm -f ~/rl/valcombresults/*.csv

sbatch ~/rl/lawson-bandits/fit_true_nn.sh
#wait

#module load R
#Rscipt ~/combine.R
