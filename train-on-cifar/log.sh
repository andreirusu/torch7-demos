#!/bin/bash -e 

while true ; do 
    date >> log.txt 
    for run in "$@"; do 
        echo -n $run' ' >> log.txt  
        OMP_NUM_THREADS=1 torch trainer.lua -threads 1 -network $run/cifar.net -test | tail -n 3 | head -n 1 >> log.txt
    done
    echo "" >> log.txt
    sleep 300
done

