#!/bin/bash

rm -f ~/slurmout/*
rm -rf ~/rl/valneuralnets/
mkdir ~/rl/valneuralnets
#rm -f ~/rl/valcombresults/*.csv
rm -rf ~/rl/neuralnetscales/
mkdir ~/rl/neuralnetscales

sbatch ~/rl/lawson-bandits/fitnn.sh
#wait

#module load R
#Rscipt ~/combine.R
