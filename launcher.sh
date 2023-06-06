#! /bin/bash
for DATA in FIX_electrode_BLP FIX_source_BLP
do
    for i in {1..9}
    do 
        DIRECTORY=/mnt/DATA/NonLinearMI/${DATA}_band${i}_bin8
        echo $i $DIRECTORY
        if [[ ! -d "$DIRECTORY" ]]
        then
            mkdir "$DIRECTORY"
            cp /mnt/DATA/NonLinearMI/${DATA}_band1_bin8/[nt]*npy "$DIRECTORY"
        fi
        /home/raffaelli/nonLinearity/cli.py -r $i -d "${DATA}" -b 8 -S -c config.ini; 
    done
done