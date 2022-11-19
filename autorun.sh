#!/bin/bash

rm -f ./running.log

if [ $# -lt 2 ]
then
        printf "Please input 'run type' and 'validation fold number' parameters!\n"
        exit 255
fi

mode=`echo $1 | tr '[A-Z]' '[a-z]'`
fold=$2

if [ "x${mode}" = "xt" ]; then
        mode="T"
elif [ "x${mode}" = "xi" ]; then
        mode="I"
elif [ "x${mode}" = "xe" ]; then
        mode="E"
else
        printf "mode error!!!! It should be 'T' or 'I'. \n"
        exit 256
fi

printf "$# parameters:: mode==${mode} val_fold=${fold}\n"
#printf "python ./Whale_Pytorch_Train.py ${mode} ${fold}"
python ./Whale_Pytorch_Train.py ${mode} ${fold}
cp ./running.log ./running.log-fold-${fold}

