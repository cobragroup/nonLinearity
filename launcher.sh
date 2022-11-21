#! /bin/bash

for i in $(cat ../list.txt) 
do 
    DIRECTORY=/home/raffaelli/eso245_cra_strin_${i}_bin9
    echo $i
    if [[ ! -d "$DIRECTORY" ]]
    then
        mkdir "$DIRECTORY"
        cp /home/raffaelli/eso245_cra_strin_10_bin9/[nt]*npy "$DIRECTORY"
    fi
    ./cli.py -r $i -d eso245_cra_strin; 
done