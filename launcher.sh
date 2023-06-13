#! /bin/bash
for DATA in FIX_source_BLP
do
    for i in 2 6 9
    do
        DIRECTORY=/mnt/DATA/NonLinearMI/${DATA}_band${i}_bin8
        echo $i $DIRECTORY
        if [[ ! -d "$DIRECTORY" ]]
        then
            mkdir "$DIRECTORY"
            cp /mnt/DATA/NonLinearMI/${DATA}_band1_bin8/[nt]*npy "$DIRECTORY"
        fi
        ./cli.py -r $i -d "${DATA}" -b 8 -c config.ini; 
    done
done