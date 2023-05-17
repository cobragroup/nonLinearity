#! /bin/bash

for i in {1..9}
do 
    DIRECTORY=/mnt/DATA/NonLinearMI/electrode_BLP_band${i}_bin8
    echo $i $DIRECTORY
    if [[ ! -d "$DIRECTORY" ]]
    then
        mkdir "$DIRECTORY"
        cp /mnt/DATA/NonLinearMI/electrode_BLP_band1_bin8/[nt]*npy "$DIRECTORY"
    fi
    ./cli.py -r $i -d electrode_BLP -b 8 -S; 
done