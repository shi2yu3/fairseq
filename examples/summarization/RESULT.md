# Docker
```
nvidia-docker run --rm -it -v $(pwd):/workspace --ipc=host pytorch/pytorch
```

# Official transformer

## Experimental setup in [translation](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)

> We trained the base models for a total of `100,000` steps or 12 hours. The big models were trained for `300,000` steps (3.5 days). We used the `Adam` optimizer with `β1 = 0.9`, `β2 = 0.98` and `epsilon = 10−9`. 
> We increased the learning rate linearly for the first `warmup_steps` training steps, and decreased it thereafter proportionally to the `inverse square root` of the step number. We used `warmup_steps = 4000`.

| Model | N | d_model | d_ff | h | d_k | d_v | P_drop | epsilon_ls | train steps | params |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| base | 6 | 512 | 2048 | 8 | 64 | 64 | 0.1 | 0.1 | 100K | 65x1e6 |
| big | 6 | 1024 | 4096 | 16 | 64 | 64 | 0.3 | - | 300K | 213x1e6 |

```
N: # identical layers
d_model: embedding dimension
d_ff: inner-layer dimension of feed-forward network
h: # multi-head in self attention
d_k: dimension of key in self attention
d_v: dimension of value in self attention
P_drop: dropout rate
epsilon_ls: label smoothing
```

## Experimental setup of ```base``` transformer in [summarization](https://arxiv.org/pdf/1904.01038.pdf)

> We truncate articles to `400` tokens (See et al., 2017). We use BPE with `30K` operations to form our vocabulary following Fan et al. (2018a). To evaluate, we use the standard ROUGE metric (Lin, 2004) and report ROUGE-1, ROUGE-2, and ROUGE-L. To generate summaries, we follow standard practice in `tuning the minimum output length` and `disallow repeating the same trigram` (Paulus et al., 2017).
> We also consider a configuration where we input `pre-trained` language model representations to the encoder network and this
language model was trained on `newscrawl and CNN-Dailymail`, totalling `193M` sentences.


# Philly config

```
# "environmentVariables": {
    "rootdir": "/philly/eu2/ipgsrch/yushi/fairseq",
    "datadir": "data-bin/cnndm",
    "arch": "transformer_vaswani_wmt_en_de_big",
    "modelpath": "/var/storage/shared/ipgsrch/sys/jobs/application_",
    "model": "1555486458178_6374",
    "epoch": "9"
  },

# train commandLine
python $rootdir/train.py $rootdir/$datadir -s src -t tgt --max-tokens 4000 -a $arch --share-all-embeddings --dropout 0.3 --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --weight-decay 0.0001 --lr 0.0005 --warmup-init-lr 1e-07 --warmup-updates 4000 --lr-scheduler inverse_sqrt --min-lr 1e-09 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-update 30000000 --save-dir $PHILLY_JOB_DIRECTORY

# test commandLine
python $rootdir/generate.py $rootdir/$datadir --path $modelpath$model/checkpoint$epoch.pt --batch-size 128 --beam 5 --remove-bpe --no-repeat-ngram-size 3 --print-alignment --output_dir $$PHILLY_JOB_DIRECTORY

# scoring
phillyfs=philly-fs.bash
id=1555486458178_15126
sudo bash $phillyfs -cp //philly/eu2/ipgsrch/sys/jobs/application_$id/candidate results/candidate
sudo bash $phillyfs -cp //philly/eu2/ipgsrch/sys/jobs/application_$id/gold results/gold
docker run --rm -it -v $(pwd):/workspace bertsum
pyrouge_set_rouge_path examples/summarization/BertSum/pyrouge/tools/ROUGE-1.5.5
python examples/summarization/BertSum/src/rouge.py
exit
```
| train | test | model | result |
| --- | --- | --- | --- |
| [1555486458178_6374](https://philly/#/job/eu2/ipgsrch/1555486458178_6374) | [xxx](https://philly/#/job/eu2/ipgsrch/xxx) | transformer_vaswani_wmt_en_de_big | xxx |


# Parameter tuning


## Baseline

```
lr = 0.0005
dropout = 0.3
weight-decay = 0.0001
label-smoothing = 0.1
max-update = 30000000
```
> train: [1555486458178_12550](https://philly/#/job/eu2/ipgsrch/1555486458178_12550)

| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |


## Learning rate tuning

* **lr = 0.05**
train: [1555486458178_12556](https://philly/#/job/eu2/ipgsrch/1555486458178_12556)

| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |

* **lr = 0.0003**
train: [1555486458178_13655](https://philly/#/job/eu2/ipgsrch/1555486458178_13655)

| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |

* **lr = 0.0001**
train: [1555486458178_13653](https://philly/#/job/eu2/ipgsrch/1555486458178_13653)

| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |

* **lr = 0.00009**
train: [1555486458178_13519](https://philly/#/job/eu2/ipgsrch/1555486458178_13519)
/ [1555486458178_13511](https://philly/#/job/eu2/ipgsrch/1555486458178_13511)

| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |

* **lr = 0.00007**
train: [1555486458178_13518](https://philly/#/job/eu2/ipgsrch/1555486458178_13518)
/ [1555486458178_13510](https://philly/#/job/eu2/ipgsrch/1555486458178_13510)

| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |

* **lr = 0.00005**
train: [1555486458178_13517](https://philly/#/job/eu2/ipgsrch/1555486458178_13517)
= [1555486458178_12557](https://philly/#/job/eu2/ipgsrch/1555486458178_12557)

| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |
| [1555486458178_15126](https://philly/#/job/eu2/ipgsrch/1555486458178_15126) | 9 | 22.22 | 4.47 | 15.40 |

* **lr = 0.00003**
train: [1555486458178_13516](https://philly/#/job/eu2/ipgsrch/1555486458178_13516)
/ [1555486458178_13508](https://philly/#/job/eu2/ipgsrch/1555486458178_13508)

| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |

* **lr = 0.00001**
train: [1555486458178_13515](https://philly/#/job/eu2/ipgsrch/1555486458178_13515)
/ [1555486458178_13509](https://philly/#/job/eu2/ipgsrch/1555486458178_13509)

| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |


## Dropout rate tuning

* **dropout = 0.0**
train: [1555486458178_12831](https://philly/#/job/eu2/ipgsrch/1555486458178_12831)

| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |

* **dropout = 0.1**
train: [1555486458178_12857](https://philly/#/job/eu2/ipgsrch/1555486458178_12857)

| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |

* **dropout = 0.2**
train: [1555486458178_12861](https://philly/#/job/eu2/ipgsrch/1555486458178_12861)

| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |


## Weight decay tuning

* **weight-decay = 0.00001**
train: [1555486458178_12865](https://philly/#/job/eu2/ipgsrch/1555486458178_12865)

| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |

* **weight-decay = 0.00005**
train: [1555486458178_12866](https://philly/#/job/eu2/ipgsrch/1555486458178_12866)

| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |


## Label smoothing tuning

* **label-smoothing = 0.01**
train: [1555486458178_12880](https://philly/#/job/eu2/ipgsrch/1555486458178_12880)

| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |

* **label-smoothing = 0.05**
train: [1555486458178_12881](https://philly/#/job/eu2/ipgsrch/1555486458178_12881)

| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |






