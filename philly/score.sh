#!/bin/bash

set -x

#cd `dirname $0`

phillyfs=philly/philly-fs.bash


for var in "$@"
do
    id=$var
    sudo rm -r results/*
    sudo bash $phillyfs -cp //philly/eu2/ipgsrch/sys/jobs/application_$id/candidate results/candidate
    sudo bash $phillyfs -cp //philly/eu2/ipgsrch/sys/jobs/application_$id/gold results/gold
    ls -l results/
    res=$(docker run --rm -it -v $(pwd):/workspace bertsum /bin/bash -c "pyrouge_set_rouge_path examples/summarization/BertSum/pyrouge/tools/ROUGE-1.5.5 && python examples/summarization/BertSum/src/rouge.py")
    echo $res
done

