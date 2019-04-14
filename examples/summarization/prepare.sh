#!/bin/bash
# Follow https://github.com/nlpyang/BertSum

apt-get update && apt-get install -y wget default-jre
pip install multiprocess pyrouge

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
TOOL_URL="http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip"
echo "TOOL_URL=$TOOL_URL"
TOOL_FILE=$(basename -- "$TOOL_URL")
echo "TOOL_FILE=$TOOL_FILE"
JAR_FILE="stanford-corenlp-3.9.2.jar"
echo "JAR_FILE=$JAR_FILE"

echo "$TOOL_FILE already exists, skipping download"
#if [ -f TOOL_FILE ]; then
#    echo "$TOOL_FILE already exists, skipping download"
#else
#    wget "$TOOL_URL"
#    if [ -f $TOOL_FILE ]; then
#        echo "$TOOL_URL successfully downloaded."
#    else
#        echo "$TOOL_URL not successfully downloaded."
#        exit -1
#    fi
#fi

echo "unziping $TOOL_FILE"
#if [ ${TOOL_FILE: -4} == ".tgz" ]; then
#    tar zxf $TOOL_FILE
#elif [ ${TOOL_FILE: -4} == ".tar" ]; then
#    tar xf $TOOL_FILE
#elif [ ${TOOL_FILE: -4} == ".zip" ]; then
#    unzip -o -q $TOOL_FILE
#fi

export CLASSPATH=$(pwd)/${TOOL_FILE%.*}/$JAR_FILE
echo "CLASSPATH=$CLASSPATH"

if ! [ -d "BertSum" ]; then
    git clone https://github.com/nlpyang/BertSum
    cp pytorch_pretrained_bert.py BertSum/src
fi
if ! [ -d "/workspace/logs" ]; then
    mkdir /workspace/logs
fi

for ((i=0;i<${#DATA_URLS[@]};++i)); do
    file=${DATA_FILES[i]}
    dir=${DATA_DIRS[i]}
    echo "$file already exists, skipping download"
    #if [ -f $file ]; then
    #    echo "$file already exists, skipping download"
    #else
    #    url=${DATA_URLS[i]}
    #    curl -c /tmp/cookies "$url" > /tmp/intermezzo.html
    #    curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > $file
    #    if [ -f $file ]; then
    #        echo "$url successfully downloaded."
    #    else
    #        echo "$url not successfully downloaded."
    #        exit -1
    #    fi
    #fi

    echo "unziping $file"
    #if [ ${file: -4} == ".tgz" ]; then
    #    tar zxf $file
    #elif [ ${file: -4} == ".tar" ]; then
    #    tar xf $file
    #elif [ ${file: -4} == ".zip" ]; then
    #    unzip $file
    #fi

    echo "tokenizing $file"
    if ! [ -d $dir/tokens ]; then
        mkdir $dir/tokens
    fi
    cd BertSum/src
    python preprocess.py -mode tokenize -raw_path ../../$dir/stories -save_path ../../$dir/tokens > ../../logs/$dir.log 2>&1
    cd ../..
done

