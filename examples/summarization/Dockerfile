FROM pytorch/pytorch:nightly-devel-cuda9.2-cudnn7

RUN apt-get update && apt-get install -y default-jre

RUN pip install pytorch_pretrained_bert tensorboardX multiprocess pyrouge

#RUN export CLASSPATH=$(pwd)/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar
