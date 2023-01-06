#! /bin/bash

for i in {1..9}
do 
    DIRECTORY=/mnt/DATA/NonLinearMI/source_BLP_band${i}_bin8
    echo $i $DIRECTORY
    if [[ ! -d "$DIRECTORY" ]]
    then
        mkdir "$DIRECTORY"
        cp /mnt/DATA/NonLinearMI/source_BLP_band1_bin8/[nt]*npy "$DIRECTORY"
    fi
    ./cli.py -r $i -d source_BLP -b 8; 
done