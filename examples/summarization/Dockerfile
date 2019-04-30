FROM pytorch/pytorch:nightly-runtime-cuda9.2-cudnn7

RUN apt-get update && apt-get install -y default-jre perl synaptic

RUN pip install pytorch_pretrained_bert tensorboardX multiprocess pyrouge
