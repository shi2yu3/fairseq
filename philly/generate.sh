#!/bin/bash

set -x

summary=("\n\n---- Summary: ")

for var in "$@"
do
    id=$var

    echo '{' > gen.json
    echo '  "version": "2019_04_20",' >> gen.json
    echo '    "metadata": {' >> gen.json
    echo '      "name": "fairseq",' >> gen.json
    echo '      "cluster": "eu2",' >> gen.json
    echo '      "vc": "ipgsrch",' >> gen.json
    echo '      "username": "yushi"' >> gen.json
    echo '    },' >> gen.json
    echo '    "environmentVariables": {' >> gen.json
    echo '      "rootdir": "/philly/eu2/ipgsrch/yushi/fairseq",' >> gen.json
    echo '      "datadir": "data-bin/cnndm",' >> gen.json
    echo '      "arch": "transformer_vaswani_wmt_en_de_big",' >> gen.json
    echo '      "modelpath": "/var/storage/shared/ipgsrch/sys/jobs/application_",' >> gen.json
    echo "      \"model\": \"$id\"," >> gen.json
    echo '      "epoch": "_best"' >> gen.json
    echo '    },' >> gen.json
    echo '    "resources": {' >> gen.json
    echo '      "workers": {' >> gen.json
    echo '        "type": "skuResource",' >> gen.json
    echo '        "sku": "G1",' >> gen.json
    echo '        "count": 1,' >> gen.json
    echo '        "image": "phillyregistry.azurecr.io/philly/jobs/custom/pytorch:pytorch1.0.0-py36-vcr",' >> gen.json
    echo '        "commandLine": "python $rootdir/generate.py $rootdir/$datadir --path $modelpath$model/checkpoint$epoch.pt --batch-size 64 --beam 5 --remove-bpe --no-repeat-ngram-size 3 --print-alignment --output_dir $PHILLY_JOB_DIRECTORY --min-len 60"' >> gen.json
    echo '      }' >> gen.json
    echo '    }' >> gen.json
    echo '  }' >> gen.json

    #res=$(curl -k --ntlm --user yushi:stZzy,myzy -X POST -H "Content-Type: application/json" --data @gen.json https://philly/api/jobs)
    res=$(curl -k --ntlm --user 'yushi' -X POST -H "Content-Type: application/json" --data @gen.json https://philly/api/jobs)
    #res='{"jobId":"application_1555486458178_20540"}'
    newid=$(echo $res | sed s/"{\"jobId\":\"application_"// | sed s/"\"}"//)
    summary=("${summary}\n${id} --> ${newid}")
done

summary=("${summary}\n\n\n")
echo -en $summary
