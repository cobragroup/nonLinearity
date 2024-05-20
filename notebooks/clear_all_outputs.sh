#!/bin/bash
echo "Clearing all outputs"
wd=$(pwd)
if [ "$(basename $wd)" != "notebooks" ]
then
    cd "/home/raffaelli/NonLinearity/nonLinearity/notebooks"
fi


for i in *.ipynb;
do
    jupyter nbconvert --clear-output --inplace $i
done

cd $wd
echo "...done!"