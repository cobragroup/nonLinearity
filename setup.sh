for DIRECTORY in bin cache
do
    if [[ ! -d "$DIRECTORY" ]]
    then
        mkdir "$DIRECTORY"
    fi
done

make
