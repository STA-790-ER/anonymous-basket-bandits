#!/bin/bash

rm -f ~/slurmout/*
rm -rf ~/rl/bernvalneuralnets/
mkdir ~/rl/bernvalneuralnets
rm ~/rl/bernneuralnetscales/*
#rm -f ~/rl/valcombresults/*.csv

sbatch ~/rl/lawson-bandits/bern_fitnn.sh
#wait

#module load R
#Rscipt ~/combine.R
