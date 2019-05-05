#!/bin/bash

set -x

phillyfs=philly/philly-fs.bash
cluster=wu3
vc=msrmt


for var in "$@"
do
    id=$var
    sudo rm -r results/*
    sudo bash $phillyfs -cp //philly/wu3/msrmt/sys/jobs/application_$id/candidate results/candidate
    sudo bash $phillyfs -cp //philly/wu3/msrmt/sys/jobs/application_$id/gold results/gold
    ls -l results/
    res=$(docker run --rm -it -v $(pwd):/workspace bertsum /bin/bash -c "pyrouge_set_rouge_path examples/summarization/BertSum/pyrouge/tools/ROUGE-1.5.5 && python examples/summarization/BertSum/src/rouge.py")
    echo $id
    echo $res
done

