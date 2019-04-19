#!/bin/bash


set -x


cd `dirname $0`


echo "*** Install rouge"
# download ROUGE-1.5.5
if ! [ -d "pyrouge" ]; then
    git clone https://github.com/andersjo/pyrouge
    cd pyrouge/tools/ROUGE-1.5.5/data
    rm WordNet-2.0.exc.db
    ./WordNet-2.0-Exceptions/buildExeptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db
    cd `dirname $0`
fi
apt-get update && apt-get install -y perl synaptic
pip install pyrouge
pyrouge_set_rouge_path pyrouge/tools/ROUGE-1.5.5


echo "*** Install other packages"
apt-get update && apt-get install -y wget default-jre zip
pip install torch pytorch_pretrained_bert multiprocess tensorboardX


# prepare for standford corenlp
echo "*** Dowload standford corenlp"
TOOL_URL="http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip"
TOOL_FILE=$(basename -- "$TOOL_URL")
JAR_FILE="stanford-corenlp-3.9.2.jar"
#if [ -f $TOOL_FILE ]; then
#    echo "    $TOOL_FILE already exists, skipping download"
#else
#    wget "$TOOL_URL"
#    if [ -f $TOOL_FILE ]; then
#        echo "    $TOOL_URL successfully downloaded."
#    else
#        echo "    $TOOL_URL not successfully downloaded."
#        exit -1
#    fi
#fi
#if [ ${TOOL_FILE: -4} == ".tgz" ]; then
#    tar zxf $TOOL_FILE
#elif [ ${TOOL_FILE: -4} == ".tar" ]; then
#    tar xf $TOOL_FILE
#elif [ ${TOOL_FILE: -4} == ".zip" ]; then
#    unzip -o -q $TOOL_FILE
#fi
export CLASSPATH=$(pwd)/${TOOL_FILE%.*}/$JAR_FILE


# get data processing code from bertsum
echo "*** Download data processing code"
if ! [ -d "BertSum" ]; then
    git clone https://github.com/shi2yu3/BertSum
fi


if ! [ -d "logs" ]; then
    mkdir logs
fi


# define corpora
DATA_URLS=(
    "https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ"
    "https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfM1BxdkxVaTY2bWs"
)
DATA_FILES=(
    "cnn_stories.tgz"
    "dailymail_stories.tgz"
)
DATA_DIRS=(
    "cnn"
    "dailymail"
)


# data processing
echo "*** Data processing"
for ((i=0;i<${#DATA_URLS[@]};++i)); do
    file=${DATA_FILES[i]}
    dir=${DATA_DIRS[i]}

    echo "    --- $file"
    #if [ -f $file ]; then
    #    echo "    $file already exists, skipping download"
    #else
    #    url=${DATA_URLS[i]}
    #    curl -c /tmp/cookies "$url" > /tmp/intermezzo.html
    #    curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > $file
    #    if [ -f $file ]; then
    #        echo "    $url successfully downloaded."
    #    else
    #        echo "    $url not successfully downloaded."
    #        exit -1
    #    fi
    #fi

    #if [ ${file: -4} == ".tgz" ]; then
    #    tar zxf $file
    #elif [ ${file: -4} == ".tar" ]; then
    #    tar xf $file
    #elif [ ${file: -4} == ".zip" ]; then
    #    unzip $file
    #fi

    cd BertSum/src

    echo "    fixing missing period in $file"
    python3 preprocess.py -mode fix_missing_period -raw_path ../../$dir/stories -save_path ../../$dir/period_fixed -log_file "" > ../../logs/${dir}_fix.log 2>&1

    echo "    tokenizing $file"
    python3 preprocess.py -mode tokenize -raw_path ../../$dir/period_fixed -save_path ../../$dir/tokens -log_file "" > ../../logs/${dir}_tok.log 2>&1

    echo "    splitting $file"
    python3 preprocess.py -mode format_to_lines -raw_path ../../$dir/tokens -save_path ../../$dir/split/$dir -map_path ../urls -log_file "" > ../../logs/${dir}_split.log 2>&1

    echo "    analyzing $file"
    python3 preprocess.py -mode analysis -raw_path ../../$dir/split -save_path ../../$dir/analysis -oracle_mode greedy -n_cpus 4 -lower -log_file ""  > ../../logs/${dir}_analysis.log 2>&1

    echo "    labeling $file"
    python3 preprocess.py -mode analysis -raw_path ../../$dir/split -save_path ../../$dir/analysis -oracle_mode greedy -n_cpus 4 -lower -log_file ""  > ../../logs/${dir}_analysis.log 2>&1

    cd ../..
done

