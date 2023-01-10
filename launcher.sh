#! /bin/bash

for i in {1..9}
do 
    DIRECTORY=/mnt/DATA/NonLinearMI/EEG_bands_band${i}_bin50
    echo $i $DIRECTORY
    if [[ ! -d "$DIRECTORY" ]]
    then
        mkdir "$DIRECTORY"
        cp /mnt/DATA/NonLinearMI/EEG_bands_band1_bin50/[nt]*npy "$DIRECTORY"
    fi
    ./cli.py -r $i -d EEG_bands -b 50; 
done