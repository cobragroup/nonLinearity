#! /bin/bash

for i in {1..9}
do 
    DIRECTORY=/mnt/DATA/NonLinearMI/iEEG_part_band${i}_bin18
    echo $i $DIRECTORY
    if [[ ! -d "$DIRECTORY" ]]
    then
        mkdir "$DIRECTORY"
        cp /mnt/DATA/NonLinearMI/iEEG_part_band1_bin18/[nt]*npy "$DIRECTORY"
    fi
    ./cli.py -r $i -d iEEG -b 18 -S; 
done