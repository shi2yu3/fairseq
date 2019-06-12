do# Table of Contents
1. [Docker](#docker)
1. [Official transformer](#official-transformer)
    1. [Experimental setup in translation paper](#experimental-setup-in-translation-paper)
    1. [Experimental setup of base transformer in summarization paper](#experimental-setup-of-base-transformer-in-summarization-paper)
1. [Philly config](#philly-config)
    1. [environmentVariables](#environmentvariables)
    1. [train commandLine](#train-commandline)
    1. [generate commandLine](#generate-commandline)
    1. [local scoring](#local-scoring)
1. [Parameter tuning](#parameter-tuning)
    1. [Baseline](#baseline)
    1. [Learning rate tuning](#learning-rate-tuning)
    1. [Dropout rate tuning](#dropout-rate-tuning)
    1. [Weight decay tuning](#weight_decay-tuning)
    1. [Label smoothing tuning](#label_smoothing-tuning)
    1. [Minimum length tunning](#minimum-length-tunning)


# Docker
```
nvidia-docker run --rm -it -v $(pwd):/workspace --ipc=host pytorch/pytorch
```

# Official transformer

## Experimental setup in [translation paper](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)

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

## Experimental setup of ```base``` transformer in [summarization paper](https://arxiv.org/pdf/1904.01038.pdf)

> We truncate articles to `400` tokens (See et al., 2017). We use BPE with `30K` operations to form our vocabulary following Fan et al. (2018a). To evaluate, we use the standard ROUGE metric (Lin, 2004) and report ROUGE-1, ROUGE-2, and ROUGE-L. To generate summaries, we follow standard practice in `tuning the minimum output length` and `disallow repeating the same trigram` (Paulus et al., 2017).
> We also consider a configuration where we input `pre-trained` language model representations to the encoder network and this
language model was trained on `newscrawl and CNN-Dailymail`, totalling `193M` sentences.


# Philly config

## environmentVariables
```
"environmentVariables": {
    "rootdir": "/philly/eu2/ipgsrch/yushi/fairseq",
    "datadir": "data-bin/cnndm",
    "arch": "transformer_vaswani_wmt_en_de_big",
    "modelpath": "/var/storage/shared/ipgsrch/sys/jobs/application_",
    "model": "1555486458178_6374",
    "epoch": "9"
  },
```

## train commandLine
```
python $rootdir/train.py $rootdir/$datadir -s src -t tgt --max-tokens 4000 -a $arch --share-all-embeddings --dropout 0.3 --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --weight-decay 0.0001 --lr 0.0005 --warmup-init-lr 1e-07 --warmup-updates 4000 --lr-scheduler inverse_sqrt --min-lr 1e-09 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-update 30000000 --max-epoch 20 --save-dir $PHILLY_JOB_DIRECTORY
```

## generate commandLine
```
python $rootdir/generate.py $rootdir/$datadir --path $modelpath$model/checkpoint$epoch.pt --batch-size 64 --beam 5 --remove-bpe --no-repeat-ngram-size 3 --print-alignment --output_dir $PHILLY_JOB_DIRECTORY --min-len 60
```

## local scoring
```
bash philly/score.sh <id>...
```


# Parameter tuning


## Default

> hyper parameter
```
--lr 0.0005
--dropout 0.3
--weight_decay 0.0001
--label_smoothing 0.1
--max_update 30000000
--min-len 1
```
> train [1555486458178_12550](https://philly/#/job/eu2/ipgsrch/1555486458178_12550)
> | last epoch 21
> | best epoch 10
> | best loss 11.4439


## Minimum length tunning

> model [1555486458178_12557](https://philly/#/job/eu2/ipgsrch/1555486458178_12557)
> | epoch 14

| test | --min-len | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |
| [1555486458178_15953](https://philly/#/job/eu2/ipgsrch/1555486458178_15953) | 10 | 21.00 | 3.48 | 13.99 |
| [1555486458178_15954](https://philly/#/job/eu2/ipgsrch/1555486458178_15954) | 20 | 21.17 | 3.50 | 14.07 |
| [1555486458178_15955](https://philly/#/job/eu2/ipgsrch/1555486458178_15955) | 30 | 21.58 | 3.57 | 14.26 |
| [1555486458178_15956](https://philly/#/job/eu2/ipgsrch/1555486458178_15956) | 40 | 22.12 | 3.69 | 14.46 |
| [1555486458178_15958](https://philly/#/job/eu2/ipgsrch/1555486458178_15958) | 50 | 22.48 | 3.84 | 14.54 |
| [**1555486458178_18095**](https://philly/#/job/eu2/ipgsrch/1555486458178_18095) | **60 (best)** | **22.50** | **3.91** | **14.50** |
| [1555486458178_18096](https://philly/#/job/eu2/ipgsrch/1555486458178_18096) | 70 | 22.35 | 3.95 | 14.34 |
| [1555486458178_18097](https://philly/#/job/eu2/ipgsrch/1555486458178_18097) | 80 | 22.04 | 3.97 | 14.09 |
| [1555486458178_18098](https://philly/#/job/eu2/ipgsrch/1555486458178_18098) | 90 | 21.65 | 3.95 | 13.81 |
| [1555486458178_18099](https://philly/#/job/eu2/ipgsrch/1555486458178_18099) | 100 | 21.20 | 3.94 | 13.52 |

We use **```--min-len 60```** in testing hereafter

## Learning rate tuning

### Single job tuning

> hyper parameter
```
--lr 0.0005
--warmup-updates 40000
--max-update 40000
```
> train: [1555486458178_17334](https://philly/#/job/eu2/ipgsrch/1555486458178_17334)
> | min loss happend at lr 0.000375037

### Separate tuning

exp ```e9172c0d```

| --lr | train | last epoch | best epoch | best loss | test | r-1 | r-2 | r-l |
| --- | --- | --- | --- | ---: | ---| ---| ---| ---|
| 0.05000 | [1555486458178_12556](https://philly/#/job/eu2/ipgsrch/1555486458178_12556) | 21 | 13 | 11.37 | [1555486458178_20537](https://philly/#/job/eu2/ipgsrch/1555486458178_20537) | 21.23 | 0.24 | 12.19 |
| 0.00030 | [1555486458178_13655](https://philly/#/job/eu2/ipgsrch/1555486458178_13655) | 20 | 15 | 11.42 | [1555486458178_20561](https://philly/#/job/eu2/ipgsrch/1555486458178_20561) | 21.78 | 0.40 | 13.21 | 
| 0.00010 | [1555486458178_13653](https://philly/#/job/eu2/ipgsrch/1555486458178_13653) | 32 | 32 | 7.92 | [1555486458178_23280](https://philly/#/job/eu2/ipgsrch/1555486458178_23280) | 24.41 | 4.49 | 15.41 |
| 0.00009 | [1555486458178_13519](https://philly/#/job/eu2/ipgsrch/1555486458178_13519) | 20 | 14 | 8.66 | [1555486458178_20538](https://philly/#/job/eu2/ipgsrch/1555486458178_20538) | 19.64 | 3.28 | 13.60 |
| 0.00007 | [1555486458178_13518](https://philly/#/job/eu2/ipgsrch/1555486458178_13518) | 20 | 20 | 8.43 | [1555486458178_20540](https://philly/#/job/eu2/ipgsrch/1555486458178_20540) | 20.83 | 3.40 | 13.73 |
| 0.00005 | [1555486458178_12557](https://philly/#/job/eu2/ipgsrch/1555486458178_12557) | 20 | 18 | 8.24 | [1555486458178_20541](https://philly/#/job/eu2/ipgsrch/1555486458178_20541) | 22.52 | 3.79 | 14.38 |
| 0.00003 | [1555486458178_13516](https://philly/#/job/eu2/ipgsrch/1555486458178_13516) | 20 | 20 | 8.10 | [1555486458178_20542](https://philly/#/job/eu2/ipgsrch/1555486458178_20542) | 25.54 | 5.28 | 16.60 |
| 0.00001 | [1555486458178_13515](https://philly/#/job/eu2/ipgsrch/1555486458178_13515) | 20 | 20 | 8.48 | [1555486458178_20543](https://philly/#/job/eu2/ipgsrch/1555486458178_20543) | 24.81 | 5.23 | 16.56 |

### Turn off regularization and lr scheduler

exp ```1529fd0e```

| --lr | train | epoch | loss | test | r-1 | r-2 | r-3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5e-06 | [1553675282044_3010](https://philly/#/job/wu3/msrmt/1553675282044_3010) | 1 | 8.130 | [1553675282044_3069](https://philly/#/job/wu3/msrmt/1553675282044_3069) | 22.10 | 4.22 | 15.22 |
| 5e-06 | [1553675282044_3010](https://philly/#/job/wu3/msrmt/1553675282044_3010) | 2 | 7.432 | [1553675282044_3070](https://philly/#/job/wu3/msrmt/1553675282044_3070) | 24.27 | 5.14 | 16.23 |
| 5e-06 | [1553675282044_3010](https://philly/#/job/wu3/msrmt/1553675282044_3010) | 3 | 6.980 | [1553675282044_3071](https://philly/#/job/wu3/msrmt/1553675282044_3071) | 27.37 | 6.25 | 17.74 |
| 5e-06 | [1553675282044_3010](https://philly/#/job/wu3/msrmt/1553675282044_3010) | 4 | 6.652 | [1553675282044_3072](https://philly/#/job/wu3/msrmt/1553675282044_3072) | 28.49 | 6.92 | 18.51 |
| 5e-06 | [1553675282044_3010](https://philly/#/job/wu3/msrmt/1553675282044_3010) | 5 | 6.462 | [1553675282044_3129](https://philly/#/job/wu3/msrmt/1553675282044_3129) | 29.93 | 7.53 | 19.18 |
| 5e-06 | [1553675282044_3010](https://philly/#/job/wu3/msrmt/1553675282044_3010) | 6 | 6.269 | [1553675282044_3130](https://philly/#/job/wu3/msrmt/1553675282044_3130) | 30.97 | 8.09 | 19.63 |
| 5e-06 | [1553675282044_3010](https://philly/#/job/wu3/msrmt/1553675282044_3010) | 7 | 6.117 | [1553675282044_3131](https://philly/#/job/wu3/msrmt/1553675282044_3131) | 31.81 | 8.38 | 19.86 |
| 5e-06 | [1553675282044_3010](https://philly/#/job/wu3/msrmt/1553675282044_3010) | 8 | 6.029 | [1553675282044_3132](https://philly/#/job/wu3/msrmt/1553675282044_3132) | 31.70 | 8.27 | 19.87 |
| 5e-06 | [1553675282044_3010](https://philly/#/job/wu3/msrmt/1553675282044_3010) | 9 | 5.943 | [1553675282044_3133](https://philly/#/job/wu3/msrmt/1553675282044_3133) | 32.62 | 8.56 | 20.18 |
| 5e-06 | [1553675282044_3010](https://philly/#/job/wu3/msrmt/1553675282044_3010) | 10 | 5.879 | [1553675282044_3134](https://philly/#/job/wu3/msrmt/1553675282044_3134) | 33.09 | 8.90 | 20.38 |
| 5e-06 | [1553675282044_3010](https://philly/#/job/wu3/msrmt/1553675282044_3010) | 11 | 5.836 | [1553675282044_3135](https://philly/#/job/wu3/msrmt/1553675282044_3135) | 33.19 | 8.88 | 20.38 |
| 5e-06 | [1553675282044_3010](https://philly/#/job/wu3/msrmt/1553675282044_3010) | 12 | 5.799 | [1553675282044_3136](https://philly/#/job/wu3/msrmt/1553675282044_3136) | 33.53 | 9.17 | 20.58 |
| 5e-06 | [1553675282044_3010](https://philly/#/job/wu3/msrmt/1553675282044_3010) | 13 | 5.790 | [1553675282044_3137](https://philly/#/job/wu3/msrmt/1553675282044_3137) | 33.24 | 9.06 | 20.47 |
| 5e-06 | [1553675282044_3010](https://philly/#/job/wu3/msrmt/1553675282044_3010) | 14 | 5.777 | [1553675282044_3198](https://philly/#/job/wu3/msrmt/1553675282044_3198) | 33.62 | 9.12 | 20.55 |
| 5e-06 | [1553675282044_3010](https://philly/#/job/wu3/msrmt/1553675282044_3010) | 15 | 5.782 | [1553675282044_3199](https://philly/#/job/wu3/msrmt/1553675282044_3199) | 33.71 | 9.30 | 20.58 |
| 5e-06 | [1553675282044_3010](https://philly/#/job/wu3/msrmt/1553675282044_3010) | 16 | 5.788 | [1553675282044_3200](https://philly/#/job/wu3/msrmt/1553675282044_3200) | 33.74 | 9.29 | 20.56 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-05 | [1553675282044_3018](https://philly/#/job/wu3/msrmt/1553675282044_3018) | 1 | 7.838 | [1553675282044_3073](https://philly/#/job/wu3/msrmt/1553675282044_3073) | 22.64 | 4.39 | 15.26 |
| 1e-05 | [1553675282044_3018](https://philly/#/job/wu3/msrmt/1553675282044_3018) | 2 | 7.040 | [1553675282044_3074](https://philly/#/job/wu3/msrmt/1553675282044_3074) | 25.15 | 5.28 | 16.51 |
| 1e-05 | [1553675282044_3018](https://philly/#/job/wu3/msrmt/1553675282044_3018) | 3 | 6.557 | [1553675282044_3075](https://philly/#/job/wu3/msrmt/1553675282044_3075) | 27.94 | 6.31 | 17.87 |
| 1e-05 | [1553675282044_3018](https://philly/#/job/wu3/msrmt/1553675282044_3018) | 4 | 6.233 | [1553675282044_3076](https://philly/#/job/wu3/msrmt/1553675282044_3076) | 28.93 | 6.94 | 18.59 |
| 1e-05 | [1553675282044_3018](https://philly/#/job/wu3/msrmt/1553675282044_3018) | 5 | 6.062 | [1553675282044_3138](https://philly/#/job/wu3/msrmt/1553675282044_3138) | 30.23 | 7.61 | 19.17 |
| 1e-05 | [1553675282044_3018](https://philly/#/job/wu3/msrmt/1553675282044_3018) | 6 | 5.926 | [1553675282044_3139](https://philly/#/job/wu3/msrmt/1553675282044_3139) | 30.73 | 7.83 | 19.42 |
| 1e-05 | [1553675282044_3018](https://philly/#/job/wu3/msrmt/1553675282044_3018) | 7 | 5.817 | [1553675282044_3140](https://philly/#/job/wu3/msrmt/1553675282044_3140) | 31.73 | 8.26 | 19.78 |
| 1e-05 | [1553675282044_3018](https://philly/#/job/wu3/msrmt/1553675282044_3018) | 8 | 5.783 | [1553675282044_3141](https://philly/#/job/wu3/msrmt/1553675282044_3141) | 31.93 | 8.29 | 19.92 |
| 1e-05 | [1553675282044_3018](https://philly/#/job/wu3/msrmt/1553675282044_3018) | 9 | 5.768 | [1553675282044_3142](https://philly/#/job/wu3/msrmt/1553675282044_3142) | 32.57 | 8.54 | 20.07 |
| 1e-05 | [1553675282044_3018](https://philly/#/job/wu3/msrmt/1553675282044_3018) | 10 | 5.767 | [1553675282044_3143](https://philly/#/job/wu3/msrmt/1553675282044_3143) | 32.82 | 8.65 | 20.13 |
| 1e-05 | [1553675282044_3018](https://philly/#/job/wu3/msrmt/1553675282044_3018) | 11 | 5.817 | [1553675282044_3144](https://philly/#/job/wu3/msrmt/1553675282044_3144) | 33.07 | 8.58 | 20.20 |
| 1e-05 | [1553675282044_3018](https://philly/#/job/wu3/msrmt/1553675282044_3018) | 12 | 5.879 | [1553675282044_3145](https://philly/#/job/wu3/msrmt/1553675282044_3145) | 33.18 | 8.71 | 20.17 |
| 1e-05 | [1553675282044_3018](https://philly/#/job/wu3/msrmt/1553675282044_3018) | 13 | 5.990 | [1553675282044_3146](https://philly/#/job/wu3/msrmt/1553675282044_3146) | 32.92 | 8.55 | 19.99 |
| 1e-05 | [1553675282044_3018](https://philly/#/job/wu3/msrmt/1553675282044_3018) | 14 | 6.098 | [1553675282044_3201](https://philly/#/job/wu3/msrmt/1553675282044_3201) | 33.38 | 8.57 | 20.07 |
| 1e-05 | [1553675282044_3018](https://philly/#/job/wu3/msrmt/1553675282044_3018) | 15 | 6.242 | [1553675282044_3202](https://philly/#/job/wu3/msrmt/1553675282044_3202) | 33.02 | 8.34 | 19.78 |
| 1e-05 | [1553675282044_3018](https://philly/#/job/wu3/msrmt/1553675282044_3018) | 16 | 6.385 | [1553675282044_3203](https://philly/#/job/wu3/msrmt/1553675282044_3203) | 32.82 | 8.18 | 19.56 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5e-05 | [1553675282044_3019](https://philly/#/job/wu3/msrmt/1553675282044_3019) | 1 | 7.483 | [1553675282044_3077](https://philly/#/job/wu3/msrmt/1553675282044_3077) | 18.40 | 2.42 | 12.29 |
| 5e-05 | [1553675282044_3019](https://philly/#/job/wu3/msrmt/1553675282044_3019) | 2 | 7.027 | [1553675282044_3078](https://philly/#/job/wu3/msrmt/1553675282044_3078) | 18.36 | 2.36 | 12.26 |
| 5e-05 | [1553675282044_3019](https://philly/#/job/wu3/msrmt/1553675282044_3019) | 3 | 6.839 | [1553675282044_3079](https://philly/#/job/wu3/msrmt/1553675282044_3079) | 19.00 | 2.61 | 12.52 |
| 5e-05 | [1553675282044_3019](https://philly/#/job/wu3/msrmt/1553675282044_3019) | 4 | 6.738 | [1553675282044_3080](https://philly/#/job/wu3/msrmt/1553675282044_3080) | 19.09 | 2.57 | 12.75 |
| 5e-05 | [1553675282044_3019](https://philly/#/job/wu3/msrmt/1553675282044_3019) | 5 | 6.738 | [1553675282044_3147](https://philly/#/job/wu3/msrmt/1553675282044_3147) | 18.92 | 2.51 | 12.52 |
| 5e-05 | [1553675282044_3019](https://philly/#/job/wu3/msrmt/1553675282044_3019) | 6 | 6.758 | [1553675282044_3148](https://philly/#/job/wu3/msrmt/1553675282044_3148) | 18.85 | 2.54 | 12.61 |
| 5e-05 | [1553675282044_3019](https://philly/#/job/wu3/msrmt/1553675282044_3019) | 7 | 6.714 | [1553675282044_3149](https://philly/#/job/wu3/msrmt/1553675282044_3149) | 18.66 | 2.24 | 12.14 |
| 5e-05 | [1553675282044_3019](https://philly/#/job/wu3/msrmt/1553675282044_3019) | 8 | 6.733 | [1553675282044_3150](https://philly/#/job/wu3/msrmt/1553675282044_3150) | 18.68 | 2.27 | 12.16 |
| 5e-05 | [1553675282044_3019](https://philly/#/job/wu3/msrmt/1553675282044_3019) | 9 | 6.714 | [1553675282044_3151](https://philly/#/job/wu3/msrmt/1553675282044_3151) | 19.31 | 2.45 | 12.47 |
| 5e-05 | [1553675282044_3019](https://philly/#/job/wu3/msrmt/1553675282044_3019) | 10 | 6.725 | [1553675282044_3152](https://philly/#/job/wu3/msrmt/1553675282044_3152) | 19.14 | 2.46 | 12.36 |
| 5e-05 | [1553675282044_3019](https://philly/#/job/wu3/msrmt/1553675282044_3019) | 11 | 6.778 | [1553675282044_3153](https://philly/#/job/wu3/msrmt/1553675282044_3153) | 19.26 | 2.40 | 12.40 |
| 5e-05 | [1553675282044_3019](https://philly/#/job/wu3/msrmt/1553675282044_3019) | 12 | 6.756 | [1553675282044_3154](https://philly/#/job/wu3/msrmt/1553675282044_3154) | 18.72 | 2.30 | 12.17 |
| 5e-05 | [1553675282044_3019](https://philly/#/job/wu3/msrmt/1553675282044_3019) | 13 | 6.812 | [1553675282044_3155](https://philly/#/job/wu3/msrmt/1553675282044_3155) | 18.99 | 2.36 | 12.29 |
| 5e-05 | [1553675282044_3019](https://philly/#/job/wu3/msrmt/1553675282044_3019) | 14 | 6.847 | [1553675282044_3204](https://philly/#/job/wu3/msrmt/1553675282044_3204) | 18.89 | 2.17 | 12.10 |
| 5e-05 | [1553675282044_3019](https://philly/#/job/wu3/msrmt/1553675282044_3019) | 15 | 6.887 | [1553675282044_3205](https://philly/#/job/wu3/msrmt/1553675282044_3205) | 17.67 | 1.81 | 11.48 |
| 5e-05 | [1553675282044_3019](https://philly/#/job/wu3/msrmt/1553675282044_3019) | 16 | 6.894 | [1553675282044_3206](https://philly/#/job/wu3/msrmt/1553675282044_3206) | 18.82 | 2.20 | 11.98 |

exp ```051e5a9a```

| --lr | train | epoch | loss | test | r-1 | r-2 | r-3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2e-05 | [1553675282044_3092](https://philly/#/job/wu3/msrmt/1553675282044_3092) | 1 | 7.626 | [1553675282044_3157](https://philly/#/job/wu3/msrmt/1553675282044_3157) | 21.41 | 3.74 | 14.28 |
| 2e-05 | [1553675282044_3092](https://philly/#/job/wu3/msrmt/1553675282044_3092) | 2 | 6.848 | [1553675282044_3158](https://philly/#/job/wu3/msrmt/1553675282044_3158) | 25.29 | 5.19 | 16.41 |
| 2e-05 | [1553675282044_3092](https://philly/#/job/wu3/msrmt/1553675282044_3092) | 3 | 6.264 | [1553675282044_3159](https://philly/#/job/wu3/msrmt/1553675282044_3159) | 28.76 | 6.58 | 18.16 |
| 2e-05 | [1553675282044_3092](https://philly/#/job/wu3/msrmt/1553675282044_3092) | 4 | 5.928 | [1553675282044_3160](https://philly/#/job/wu3/msrmt/1553675282044_3160) | 29.95 | 7.47 | 18.97 |
| 2e-05 | [1553675282044_3092](https://philly/#/job/wu3/msrmt/1553675282044_3092) | 5 | 5.770 | [1553675282044_3161](https://philly/#/job/wu3/msrmt/1553675282044_3161) | 31.39 | 8.20 | 19.74 |
| 2e-05 | [1553675282044_3092](https://philly/#/job/wu3/msrmt/1553675282044_3092) | 6 | 5.614 | [1553675282044_3162](https://philly/#/job/wu3/msrmt/1553675282044_3162) | 32.45 | 8.80 | 20.30 |
| 2e-05 | [1553675282044_3092](https://philly/#/job/wu3/msrmt/1553675282044_3092) | 7 | 5.486 | [1553675282044_3163](https://philly/#/job/wu3/msrmt/1553675282044_3163) | 33.45 | 9.60 | 20.72 |
| 2e-05 | [1553675282044_3092](https://philly/#/job/wu3/msrmt/1553675282044_3092) | 8 | 5.414 | [1553675282044_3189](https://philly/#/job/wu3/msrmt/1553675282044_3189) | 33.75 | 9.79 | 20.91 |
| 2e-05 | [1553675282044_3092](https://philly/#/job/wu3/msrmt/1553675282044_3092) | 9 | 5.383 | [1553675282044_3190](https://philly/#/job/wu3/msrmt/1553675282044_3190) | 34.42 | 10.30 | 21.24 |
| 2e-05 | [1553675282044_3092](https://philly/#/job/wu3/msrmt/1553675282044_3092) | 10 | 5.409 | [1553675282044_3191](https://philly/#/job/wu3/msrmt/1553675282044_3191) | 34.93 | 10.83 | 21.50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3e-05 | [1553675282044_3093](https://philly/#/job/wu3/msrmt/1553675282044_3093) | 1 | 7.547 | [1553675282044_3164](https://philly/#/job/wu3/msrmt/1553675282044_3164) | 20.00 | 3.02 | 13.24 |
| 3e-05 | [1553675282044_3093](https://philly/#/job/wu3/msrmt/1553675282044_3093) | 2 | 7.089 | [1553675282044_3165](https://philly/#/job/wu3/msrmt/1553675282044_3165) | 21.29 | 3.43 | 13.89 |
| 3e-05 | [1553675282044_3093](https://philly/#/job/wu3/msrmt/1553675282044_3093) | 3 | 6.853 | [1553675282044_3166](https://philly/#/job/wu3/msrmt/1553675282044_3166) | 21.89 | 3.61 | 13.97 |
| 3e-05 | [1553675282044_3093](https://philly/#/job/wu3/msrmt/1553675282044_3093) | 4 | 6.739 | [1553675282044_3167](https://philly/#/job/wu3/msrmt/1553675282044_3167) | 22.23 | 3.83 | 14.40 |
| 3e-05 | [1553675282044_3093](https://philly/#/job/wu3/msrmt/1553675282044_3093) | 5 | 6.731 | [1553675282044_3168](https://philly/#/job/wu3/msrmt/1553675282044_3168) | 22.65 | 3.93 | 14.47 |
| 3e-05 | [1553675282044_3093](https://philly/#/job/wu3/msrmt/1553675282044_3093) | 6 | 6.697 | [1553675282044_3169](https://philly/#/job/wu3/msrmt/1553675282044_3169) | 23.14 | 4.12 | 14.81 |
| 3e-05 | [1553675282044_3093](https://philly/#/job/wu3/msrmt/1553675282044_3093) | 7 | 6.697 | [1553675282044_3170](https://philly/#/job/wu3/msrmt/1553675282044_3170) | 23.17 | 4.04 | 14.70 |
| 3e-05 | [1553675282044_3093](https://philly/#/job/wu3/msrmt/1553675282044_3093) | 8 | 6.732 | [1553675282044_3192](https://philly/#/job/wu3/msrmt/1553675282044_3192) | 23.54 | 4.11 | 14.91 |
| 3e-05 | [1553675282044_3093](https://philly/#/job/wu3/msrmt/1553675282044_3093) | 9 | 6.751 | [1553675282044_3193](https://philly/#/job/wu3/msrmt/1553675282044_3193) | 23.86 | 4.09 | 14.94 |
| 3e-05 | [1553675282044_3093](https://philly/#/job/wu3/msrmt/1553675282044_3093) | 10 | 6.781 | [1553675282044_3194](https://philly/#/job/wu3/msrmt/1553675282044_3194) | 23.25 | 3.91 | 14.50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4e-05 | [1553675282044_3094](https://philly/#/job/wu3/msrmt/1553675282044_3094) | 1 | 7.502 | [1553675282044_3171](https://philly/#/job/wu3/msrmt/1553675282044_3171) | 19.05 | 2.63 | 12.67 |
| 4e-05 | [1553675282044_3094](https://philly/#/job/wu3/msrmt/1553675282044_3094) | 2 | 7.026 | [1553675282044_3172](https://philly/#/job/wu3/msrmt/1553675282044_3172) | 20.12 | 2.97 | 13.26 |
| 4e-05 | [1553675282044_3094](https://philly/#/job/wu3/msrmt/1553675282044_3094) | 3 | 6.839 | [1553675282044_3173](https://philly/#/job/wu3/msrmt/1553675282044_3173) | 20.34 | 3.10 | 13.27 |
| 4e-05 | [1553675282044_3094](https://philly/#/job/wu3/msrmt/1553675282044_3094) | 4 | 6.715 | [1553675282044_3174](https://philly/#/job/wu3/msrmt/1553675282044_3174) | 20.31 | 3.11 | 13.46 |
| 4e-05 | [1553675282044_3094](https://philly/#/job/wu3/msrmt/1553675282044_3094) | 5 | 6.713 | [1553675282044_3175](https://philly/#/job/wu3/msrmt/1553675282044_3175) | 20.45 | 3.08 | 13.35 |
| 4e-05 | [1553675282044_3094](https://philly/#/job/wu3/msrmt/1553675282044_3094) | 6 | 6.701 | [1553675282044_3176](https://philly/#/job/wu3/msrmt/1553675282044_3176) | 20.96 | 3.32 | 13.75 |
| 4e-05 | [1553675282044_3094](https://philly/#/job/wu3/msrmt/1553675282044_3094) | 7 | 6.720 | [1553675282044_3177](https://philly/#/job/wu3/msrmt/1553675282044_3177) | 20.39 | 2.98 | 13.19 |
| 4e-05 | [1553675282044_3094](https://philly/#/job/wu3/msrmt/1553675282044_3094) | 8 | 6.719 | [1553675282044_3195](https://philly/#/job/wu3/msrmt/1553675282044_3195) | 20.66 | 2.99 | 13.36 |
| 4e-05 | [1553675282044_3094](https://philly/#/job/wu3/msrmt/1553675282044_3094) | 9 | 6.709 | [1553675282044_3196](https://philly/#/job/wu3/msrmt/1553675282044_3196) | 20.37 | 2.79 | 12.99 |
| 4e-05 | [1553675282044_3094](https://philly/#/job/wu3/msrmt/1553675282044_3094) | 10 | 6.757 | [1553675282044_3197](https://philly/#/job/wu3/msrmt/1553675282044_3197) | 19.62 | 2.70 | 12.71 |


## Bayesian optimization

### fixed lr scheduler

exp ```11853456```
```
nohup python3.6 -u bo.py --num_new_jobs 6 --num_rounds 10 --port 9001 bayesian/11853456/ > 11853456.log &
python gen.py --epoch _best --min_len "30 35 40 45 50 55 60 65 70" 1553675282044_3404
bash philly/score.sh 1553675282044_3986 1553675282044_3987 1553675282044_3499 1553675282044_3538 1553675282044_3502 1553675282044_3539 1553675282044_3490 1553675282044_3540 1553675282044_3503
```

| train | epoch | loss | test | min_len | r-1 | r-2 | r-3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [1553675282044_3404](https://philly/#/job/wu3/msrmt/1553675282044_3404) | _best | 4.655 | [1553675282044_3986](https://philly/#/job/wu3/msrmt/1553675282044_3986) | 30 | 36.32 | 14.45 | 24.23 |
| [1553675282044_3404](https://philly/#/job/wu3/msrmt/1553675282044_3404) | _best | 4.655 | [1553675282044_3987](https://philly/#/job/wu3/msrmt/1553675282044_3987) | 35 | 36.40 | 14.49 | **24.25** |
| [1553675282044_3404](https://philly/#/job/wu3/msrmt/1553675282044_3404) | _best | 4.655 | [1553675282044_3499](https://philly/#/job/wu3/msrmt/1553675282044_3499) | 40 | 36.49 | 14.49 | 24.23 |
| [1553675282044_3404](https://philly/#/job/wu3/msrmt/1553675282044_3404) | _best | 4.655 | [1553675282044_3538](https://philly/#/job/wu3/msrmt/1553675282044_3538) | 45 | 36.61 | **14.50** | 24.18 |
| [1553675282044_3404](https://philly/#/job/wu3/msrmt/1553675282044_3404) | _best | 4.655 | [1553675282044_3502](https://philly/#/job/wu3/msrmt/1553675282044_3502) | 50 | 36.70 | 14.48 | 24.09 |
| [1553675282044_3404](https://philly/#/job/wu3/msrmt/1553675282044_3404) | _best | 4.655 | [1553675282044_3539](https://philly/#/job/wu3/msrmt/1553675282044_3539) | 55 | 36.73 | 14.46 | 23.98 |
| [1553675282044_3404](https://philly/#/job/wu3/msrmt/1553675282044_3404) | _best | 4.655 | [1553675282044_3490](https://philly/#/job/wu3/msrmt/1553675282044_3490) | 60 | **36.74** | 14.46 | 23.84 |
| [1553675282044_3404](https://philly/#/job/wu3/msrmt/1553675282044_3404) | _best | 4.655 | [1553675282044_3540](https://philly/#/job/wu3/msrmt/1553675282044_3540) | 65 | 36.65 | 14.37 | 23.63 |
| [1553675282044_3404](https://philly/#/job/wu3/msrmt/1553675282044_3404) | _best | 4.655 | [1553675282044_3503](https://philly/#/job/wu3/msrmt/1553675282044_3503) | 70 | 36.51 | 14.28 | 23.39 |

exp ```485a920e```

```
nohup python3.6 -u bo.py --num_new_jobs 6 --num_rounds 10 --port 9005 bayesian/485a920e/ > 485a920e.log &
python gen.py --epoch _best --min_len "30 35 40 45 50 55 60 65 70" 1553675282044_8686
bash philly/score.sh 1553675282044_10211 1553675282044_10212 1553675282044_10213 1553675282044_10214 1553675282044_10215 1553675282044_10216 1553675282044_10217 1553675282044_10218 1553675282044_10219
```

| train | epoch | loss | test | min_len | r-1 | r-2 | r-3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [1553675282044_8686](https://philly/#/job/wu3/msrmt/1553675282044_8686) | _best | 4.415 | [1553675282044_10211](https://philly/#/job/wu3/msrmt/1553675282044_10211) | 30 | 37.71 | 15.74 | 25.70 |
| [1553675282044_8686](https://philly/#/job/wu3/msrmt/1553675282044_8686) | _best | 4.415 | [1553675282044_10212](https://philly/#/job/wu3/msrmt/1553675282044_10212) | 35 | 37.82 | 15.77 | **25.71** |
| [1553675282044_8686](https://philly/#/job/wu3/msrmt/1553675282044_8686) | _best | 4.415 | [1553675282044_10213](https://philly/#/job/wu3/msrmt/1553675282044_10213) | 40 | 37.96 | **15.79** | 25.70 |
| [1553675282044_8686](https://philly/#/job/wu3/msrmt/1553675282044_8686) | _best | 4.415 | [1553675282044_10214](https://philly/#/job/wu3/msrmt/1553675282044_10214) | 45 | 38.08 | 15.78 | 25.65 |
| [1553675282044_8686](https://philly/#/job/wu3/msrmt/1553675282044_8686) | _best | 4.415 | [1553675282044_10215](https://philly/#/job/wu3/msrmt/1553675282044_10215) | 50 | 38.15 | 15.74 | 25.52 |
| [1553675282044_8686](https://philly/#/job/wu3/msrmt/1553675282044_8686) | _best | 4.415 | [1553675282044_10216](https://philly/#/job/wu3/msrmt/1553675282044_10216) | 55 | **38.20** | 15.72 | 25.40 |
| [1553675282044_8686](https://philly/#/job/wu3/msrmt/1553675282044_8686) | _best | 4.415 | [1553675282044_10217](https://philly/#/job/wu3/msrmt/1553675282044_10217) | 60 | 38.18 | 15.70 | 25.25 |
| [1553675282044_8686](https://philly/#/job/wu3/msrmt/1553675282044_8686) | _best | 4.415 | [1553675282044_10218](https://philly/#/job/wu3/msrmt/1553675282044_10218) | 65 | 38.11 | 15.64 | 25.04 |
| [1553675282044_8686](https://philly/#/job/wu3/msrmt/1553675282044_8686) | _best | 4.415 | [1553675282044_10219](https://philly/#/job/wu3/msrmt/1553675282044_10219) | 70 | 37.98 | 15.59 | 24.83 |

### cosine lr scheduler

exp ```07e509b8```

```
nohup python3.6 -u bo.py --num_new_jobs 6 --num_rounds 10 --port 9000 bayesian/07e509b8/ > 07e509b8.log &
python gen.py --epoch _best --min_len "30 35 40 45 50 55 60 65 70" 1553675282044_4232
bash philly/score.sh 1553675282044_10220 1553675282044_10221 1553675282044_10222 1553675282044_10223 1553675282044_10224 1553675282044_10225 1553675282044_10226 1553675282044_10227 1553675282044_10228
```

| train | epoch | loss | test | min_len | r-1 | r-2 | r-3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [1553675282044_4232](https://philly/#/job/wu3/msrmt/1553675282044_4232) | _best | 5.887 | [1553675282044_10220](https://philly/#/job/wu3/msrmt/1553675282044_10220) | 30 | 29.55 | 7.85 | 19.14 |
| [1553675282044_4232](https://philly/#/job/wu3/msrmt/1553675282044_4232) | _best | 5.887 | [1553675282044_10221](https://philly/#/job/wu3/msrmt/1553675282044_10221) | 35 | 29.66 | 7.86 | **19.16** |
| [1553675282044_4232](https://philly/#/job/wu3/msrmt/1553675282044_4232) | _best | 5.887 | [1553675282044_10222](https://philly/#/job/wu3/msrmt/1553675282044_10222) | 40 | 29.78 | **7.86** | 19.13 |
| [1553675282044_4232](https://philly/#/job/wu3/msrmt/1553675282044_4232) | _best | 5.887 | [1553675282044_10223](https://philly/#/job/wu3/msrmt/1553675282044_10223) | 45 | 29.94 | 7.86 | 19.07 |
| [1553675282044_4232](https://philly/#/job/wu3/msrmt/1553675282044_4232) | _best | 5.887 | [1553675282044_10224](https://philly/#/job/wu3/msrmt/1553675282044_10224) | 50 | **30.01** | 7.85 | 18.97 |
| [1553675282044_4232](https://philly/#/job/wu3/msrmt/1553675282044_4232) | _best | 5.887 | [1553675282044_10225](https://philly/#/job/wu3/msrmt/1553675282044_10225) | 55 | 29.97 | 7.81 | 18.83 |
| [1553675282044_4232](https://philly/#/job/wu3/msrmt/1553675282044_4232) | _best | 5.887 | [1553675282044_10226](https://philly/#/job/wu3/msrmt/1553675282044_10226) | 60 | 29.83 | 7.76 | 18.65 |
| [1553675282044_4232](https://philly/#/job/wu3/msrmt/1553675282044_4232) | _best | 5.887 | [1553675282044_10227](https://philly/#/job/wu3/msrmt/1553675282044_10227) | 65 | 29.63 | 7.69 | 18.44 |
| [1553675282044_4232](https://philly/#/job/wu3/msrmt/1553675282044_4232) | _best | 5.887 | [1553675282044_10228](https://philly/#/job/wu3/msrmt/1553675282044_10228) | 70 | 29.40 | 7.61 | 18.24 |

### inverse_sqrt lr scheduler

exp ```9394bf00```

```
nohup python3.6 -u bo.py --num_new_jobs 6 --num_rounds 10 --port 9002 bayesian/9394bf00/ > 9394bf00.log &
python gen.py --epoch _best --min_len "30 35 40 45 50 55 60 65 70" 1553675282044_3784
bash philly/score.sh 1553675282044_3982 1553675282044_3983 1553675282044_3984 1553675282044_3985 1553675282044_3964 1553675282044_3965 1553675282044_3966 1553675282044_3967
```

| train | epoch | loss | test | min_len | r-1 | r-2 | r-3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [1553675282044_3784](https://philly/#/job/wu3/msrmt/1553675282044_3784) | _best | 3.986 | [1553675282044_3982](https://philly/#/job/wu3/msrmt/1553675282044_3982) | 30 | 38.13 | 16.37 | 26.29 |
| [1553675282044_3784](https://philly/#/job/wu3/msrmt/1553675282044_3784) | _best | 3.986 | [1553675282044_3983](https://philly/#/job/wu3/msrmt/1553675282044_3983) | 35 | 38.25 | 16.40 | 26.32 |
| [1553675282044_3784](https://philly/#/job/wu3/msrmt/1553675282044_3784) | _best | 3.986 | [1553675282044_3984](https://philly/#/job/wu3/msrmt/1553675282044_3984) | 40 | 38.39 | 16.43 | **26.32** |
| [1553675282044_3784](https://philly/#/job/wu3/msrmt/1553675282044_3784) | _best | 3.986 | [1553675282044_3985](https://philly/#/job/wu3/msrmt/1553675282044_3985) | 45 | 38.62 | 16.46 | 26.30 |
| [1553675282044_3784](https://philly/#/job/wu3/msrmt/1553675282044_3784) | _best | 3.986 | [1553675282044_3964](https://philly/#/job/wu3/msrmt/1553675282044_3964) | 50 | 38.79 | **16.47** | 26.23 |
| [1553675282044_3784](https://philly/#/job/wu3/msrmt/1553675282044_3784) | _best | 3.986 | [1553675282044_3965](https://philly/#/job/wu3/msrmt/1553675282044_3965) | 55 | 38.91 | 16.46 | 26.11 |
| [1553675282044_3784](https://philly/#/job/wu3/msrmt/1553675282044_3784) | _best | 3.986 | [1553675282044_3966](https://philly/#/job/wu3/msrmt/1553675282044_3966) | 60 | **38.97** | 16.43 | 25.95 |
| [1553675282044_3784](https://philly/#/job/wu3/msrmt/1553675282044_3784) | _best | 3.986 | [1553675282044_3967](https://philly/#/job/wu3/msrmt/1553675282044_3967) | 65 | 38.90 | 16.36 | 25.73 |

exp ```ab020539```

```
nohup python3.6 -u bo.py --num_new_jobs 6 --num_rounds 10 --port 9004 bayesian/ab020539/ > ab020539.log &
python gen.py --epoch _best --min_len "30 35 40 45 50 55 60 65 70" 1553675282044_5446
bash philly/score.sh 1553675282044_10229 1553675282044_10230 1553675282044_10231 1553675282044_10232 1553675282044_10233 1553675282044_10234 1553675282044_10235 1553675282044_10236 1553675282044_10237
```
Model expired on Philly.

| train | epoch | loss | test | min_len | r-1 | r-2 | r-3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [1553675282044_5446](https://philly/#/job/wu3/msrmt/1553675282044_5446) | _best | 4.164 | [1553675282044_10229](https://philly/#/job/wu3/msrmt/1553675282044_10229) | 30 | --- | --- | --- |
| [1553675282044_5446](https://philly/#/job/wu3/msrmt/1553675282044_5446) | _best | 4.164 | [1553675282044_10230](https://philly/#/job/wu3/msrmt/1553675282044_10230) | 35 | --- | --- | --- |
| [1553675282044_5446](https://philly/#/job/wu3/msrmt/1553675282044_5446) | _best | 4.164 | [1553675282044_10231](https://philly/#/job/wu3/msrmt/1553675282044_10231) | 40 | --- | --- | --- |
| [1553675282044_5446](https://philly/#/job/wu3/msrmt/1553675282044_5446) | _best | 4.164 | [1553675282044_10232](https://philly/#/job/wu3/msrmt/1553675282044_10232) | 45 | --- | --- | --- |
| [1553675282044_5446](https://philly/#/job/wu3/msrmt/1553675282044_5446) | _best | 4.164 | [1553675282044_10233](https://philly/#/job/wu3/msrmt/1553675282044_10233) | 50 | --- | --- | --- |
| [1553675282044_5446](https://philly/#/job/wu3/msrmt/1553675282044_5446) | _best | 4.164 | [1553675282044_10234](https://philly/#/job/wu3/msrmt/1553675282044_10234) | 55 | --- | --- | --- |
| [1553675282044_5446](https://philly/#/job/wu3/msrmt/1553675282044_5446) | _best | 4.164 | [1553675282044_10235](https://philly/#/job/wu3/msrmt/1553675282044_10235) | 60 | --- | --- | --- |
| [1553675282044_5446](https://philly/#/job/wu3/msrmt/1553675282044_5446) | _best | 4.164 | [1553675282044_10236](https://philly/#/job/wu3/msrmt/1553675282044_10236) | 65 | --- | --- | --- |
| [1553675282044_5446](https://philly/#/job/wu3/msrmt/1553675282044_5446) | _best | 4.164 | [1553675282044_10237](https://philly/#/job/wu3/msrmt/1553675282044_10237) | 70 | --- | --- | --- |

### triangular lr scheduler

exp ```e3f5b02d```

```
nohup python3.6 -u bo.py --num_new_jobs 6 --num_rounds 10 --port 9003 bayesian/e3f5b02d/ > e3f5b02d.log &
python gen.py --epoch _best --min_len "30 35 40 45 50 55 60 65 70" 1553675282044_4778
bash philly/score.sh 1553675282044_10238 1553675282044_10239 1553675282044_10240 1553675282044_10241 1553675282044_10242 1553675282044_10243 1553675282044_10244 1553675282044_10245 1553675282044_10246
```
Model expired on Philly.

| train | epoch | loss | test | min_len | r-1 | r-2 | r-3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [1553675282044_4778](https://philly/#/job/wu3/msrmt/1553675282044_4778) | _best | 6.097 | [1553675282044_10238](https://philly/#/job/wu3/msrmt/1553675282044_10238) | 30 | --- | --- | --- |
| [1553675282044_4778](https://philly/#/job/wu3/msrmt/1553675282044_4778) | _best | 6.097 | [1553675282044_10239](https://philly/#/job/wu3/msrmt/1553675282044_10239) | 35 | --- | --- | --- |
| [1553675282044_4778](https://philly/#/job/wu3/msrmt/1553675282044_4778) | _best | 6.097 | [1553675282044_10240](https://philly/#/job/wu3/msrmt/1553675282044_10240) | 40 | --- | --- | --- |
| [1553675282044_4778](https://philly/#/job/wu3/msrmt/1553675282044_4778) | _best | 6.097 | [1553675282044_10241](https://philly/#/job/wu3/msrmt/1553675282044_10241) | 45 | --- | --- | --- |
| [1553675282044_4778](https://philly/#/job/wu3/msrmt/1553675282044_4778) | _best | 6.097 | [1553675282044_10242](https://philly/#/job/wu3/msrmt/1553675282044_10242) | 50 | --- | --- | --- |
| [1553675282044_4778](https://philly/#/job/wu3/msrmt/1553675282044_4778) | _best | 6.097 | [1553675282044_10243](https://philly/#/job/wu3/msrmt/1553675282044_10243) | 55 | --- | --- | --- |
| [1553675282044_4778](https://philly/#/job/wu3/msrmt/1553675282044_4778) | _best | 6.097 | [1553675282044_10244](https://philly/#/job/wu3/msrmt/1553675282044_10244) | 60 | --- | --- | --- |
| [1553675282044_4778](https://philly/#/job/wu3/msrmt/1553675282044_4778) | _best | 6.097 | [1553675282044_10245](https://philly/#/job/wu3/msrmt/1553675282044_10245) | 65 | --- | --- | --- |
| [1553675282044_4778](https://philly/#/job/wu3/msrmt/1553675282044_4778) | _best | 6.097 | [1553675282044_10246](https://philly/#/job/wu3/msrmt/1553675282044_10246) | 70 | --- | --- | --- |


## Grid search

exp ```12a8f7da```

| clip-norm | dropout | label-smoothing | lr | weight-decay | train | epoch | loss | test | r-1 | r-2 | r-3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 0.1 | 0.0 | 0.0001 | 0.0 | [1555486458178_20449](https://philly/#/job/eu2/ipgsrch/1555486458178_20449) | _best | 6.638 | [1555486458178_25501](https://philly/#/job/eu2/ipgsrch/1555486458178_25501) | 23.17 | 4.08 | 14.86 |
| 0.0 | 0.1 | 0.1 | 0.0001 | 0.0 | [1555486458178_20450](https://philly/#/job/eu2/ipgsrch/1555486458178_20450) | _best | 8.338 | [1555486458178_25502](https://philly/#/job/eu2/ipgsrch/1555486458178_25502) | 22.68 | 3.79 | 14.49 |
| 0.0 | 0.1 | 0.2 | 0.0001 | 0.0 | [1555486458178_20451](https://philly/#/job/eu2/ipgsrch/1555486458178_20451) | _best | 8.969 | [1555486458178_25503](https://philly/#/job/eu2/ipgsrch/1555486458178_25503) | 22.32 | 3.61 | 14.24 |
| 0.0 | 0.1 | 0.0 | 0.0001 | 1e-05 | [1555486458178_20452](https://philly/#/job/eu2/ipgsrch/1555486458178_20452) | _best | 6.651 | [1555486458178_25504](https://philly/#/job/eu2/ipgsrch/1555486458178_25504) | 23.20 | 3.97 | 14.84 |
| 0.0 | 0.1 | 0.1 | 0.0001 | 1e-05 | [1555486458178_20453](https://philly/#/job/eu2/ipgsrch/1555486458178_20453) | _best | 7.883 | [1555486458178_25505](https://philly/#/job/eu2/ipgsrch/1555486458178_25505) | 22.45 | 3.76 | 14.41 |
| 0.0 | 0.1 | 0.2 | 0.0001 | 1e-05 | [1555486458178_20454](https://philly/#/job/eu2/ipgsrch/1555486458178_20454) | _best | 8.969 | [1555486458178_25506](https://philly/#/job/eu2/ipgsrch/1555486458178_25506) | 22.32 | 3.61 | 14.24 |
| 0.0 | 0.1 | 0.0 | 0.0001 | 0.0001 | [1555486458178_20455](https://philly/#/job/eu2/ipgsrch/1555486458178_20455) | _best | 6.638 | [1555486458178_25507](https://philly/#/job/eu2/ipgsrch/1555486458178_25507) | 23.17 | 4.08 | 14.86 |
| 0.0 | 0.1 | 0.1 | 0.0001 | 0.0001 | [1555486458178_20456](https://philly/#/job/eu2/ipgsrch/1555486458178_20456) | _best | 7.86 | [1555486458178_25508](https://philly/#/job/eu2/ipgsrch/1555486458178_25508) | 22.52 | 3.84 | 14.41 |
| 0.0 | 0.1 | 0.2 | 0.0001 | 0.0001 | [1555486458178_20457](https://philly/#/job/eu2/ipgsrch/1555486458178_20457) | _best | 9.137 | [1555486458178_25509](https://philly/#/job/eu2/ipgsrch/1555486458178_25509) | 22.32 | 3.61 | 14.24 |
| 0.1 | 0.1 | 0.0 | 0.0001 | 0.0 | [1555486458178_20458](https://philly/#/job/eu2/ipgsrch/1555486458178_20458) | _best | 7.052 | [1555486458178_25510](https://philly/#/job/eu2/ipgsrch/1555486458178_25510) | 21.98 | 3.61 | 14.11 |
| 0.1 | 0.1 | 0.1 | 0.0001 | 0.0 | [1555486458178_20459](https://philly/#/job/eu2/ipgsrch/1555486458178_20459) | _best | 7.877 | [1555486458178_25511](https://philly/#/job/eu2/ipgsrch/1555486458178_25511) | 20.67 | 3.13 | 13.24 |
| 0.1 | 0.1 | 0.2 | 0.0001 | 0.0 | [1555486458178_20460](https://philly/#/job/eu2/ipgsrch/1555486458178_20460) | _best | 9.472 | [1555486458178_25512](https://philly/#/job/eu2/ipgsrch/1555486458178_25512) | 22.82 | 3.94 | 14.52 |
| 0.1 | 0.1 | 0.0 | 0.0001 | 1e-05 | [1555486458178_20461](https://philly/#/job/eu2/ipgsrch/1555486458178_20461) | _best | 7.052 | [1555486458178_25513](https://philly/#/job/eu2/ipgsrch/1555486458178_25513) | 21.98 | 3.61 | 14.11 |
| 0.1 | 0.1 | 0.1 | 0.0001 | 1e-05 | [1555486458178_20462](https://philly/#/job/eu2/ipgsrch/1555486458178_20462) | _best | 8.127 | [1555486458178_25514](https://philly/#/job/eu2/ipgsrch/1555486458178_25514) | 20.67 | 3.13 | 13.24 |
| 0.1 | 0.1 | 0.2 | 0.0001 | 1e-05 | [1555486458178_20463](https://philly/#/job/eu2/ipgsrch/1555486458178_20463) | _best | 9.115 | [1555486458178_25515](https://philly/#/job/eu2/ipgsrch/1555486458178_25515) | 22.82 | 3.94 | 14.52 |
| 0.1 | 0.1 | 0.0 | 0.0001 | 0.0001 | [1555486458178_20464](https://philly/#/job/eu2/ipgsrch/1555486458178_20464) | _best | 6.683 | [1555486458178_25516](https://philly/#/job/eu2/ipgsrch/1555486458178_25516) | 21.98 | 3.61 | 14.11 |
| 0.1 | 0.1 | 0.1 | 0.0001 | 0.0001 | [1555486458178_20465](https://philly/#/job/eu2/ipgsrch/1555486458178_20465) | _best | 7.877 | [1555486458178_25517](https://philly/#/job/eu2/ipgsrch/1555486458178_25517) | 20.67 | 3.13 | 13.24 |
| 0.1 | 0.1 | 0.2 | 0.0001 | 0.0001 | [1555486458178_20466](https://philly/#/job/eu2/ipgsrch/1555486458178_20466) | _best | 8.819 | [1555486458178_25518](https://philly/#/job/eu2/ipgsrch/1555486458178_25518) | 22.82 | 3.94 | 14.52 |
| 0.2 | 0.1 | 0.0 | 0.0001 | 0.0 | [1555486458178_20467](https://philly/#/job/eu2/ipgsrch/1555486458178_20467) | _best | 6.718 | [1555486458178_25519](https://philly/#/job/eu2/ipgsrch/1555486458178_25519) | 22.52 | 3.69 | 14.38 |
| 0.2 | 0.1 | 0.1 | 0.0001 | 0.0 | [1555486458178_20468](https://philly/#/job/eu2/ipgsrch/1555486458178_20468) | _best | 7.946 | [1555486458178_25520](https://philly/#/job/eu2/ipgsrch/1555486458178_25520) | 21.90 | 3.52 | 14.06 |
| 0.2 | 0.1 | 0.2 | 0.0001 | 0.0 | [1555486458178_20469](https://philly/#/job/eu2/ipgsrch/1555486458178_20469) | _best | 8.914 | [1555486458178_25521](https://philly/#/job/eu2/ipgsrch/1555486458178_25521) | 21.55 | 3.43 | 13.88 |
| 0.2 | 0.1 | 0.0 | 0.0001 | 1e-05 | [1555486458178_20470](https://philly/#/job/eu2/ipgsrch/1555486458178_20470) | _best | 6.724 | [1555486458178_25522](https://philly/#/job/eu2/ipgsrch/1555486458178_25522) | 22.54 | 3.70 | 14.40 |
| 0.2 | 0.1 | 0.1 | 0.0001 | 1e-05 | [1555486458178_20471](https://philly/#/job/eu2/ipgsrch/1555486458178_20471) | _best | 7.882 | [1555486458178_25523](https://philly/#/job/eu2/ipgsrch/1555486458178_25523) | 21.90 | 3.59 | 14.09 |
| 0.2 | 0.1 | 0.2 | 0.0001 | 1e-05 | [1555486458178_20472](https://philly/#/job/eu2/ipgsrch/1555486458178_20472) | _best | 8.914 | [1555486458178_25524](https://philly/#/job/eu2/ipgsrch/1555486458178_25524) | 21.55 | 3.43 | 13.88 |
| 0.2 | 0.1 | 0.0 | 0.0001 | 0.0001 | [1555486458178_20473](https://philly/#/job/eu2/ipgsrch/1555486458178_20473) | _best | 6.753 | [1555486458178_25525](https://philly/#/job/eu2/ipgsrch/1555486458178_25525) | 21.92 | 3.55 | 14.15 |
| 0.2 | 0.1 | 0.1 | 0.0001 | 0.0001 | [1555486458178_20474](https://philly/#/job/eu2/ipgsrch/1555486458178_20474) | _best | 7.871 | [1555486458178_25526](https://philly/#/job/eu2/ipgsrch/1555486458178_25526) | 21.90 | 3.52 | 14.06 |
| 0.2 | 0.1 | 0.2 | 0.0001 | 0.0001 | [1555486458178_20475](https://philly/#/job/eu2/ipgsrch/1555486458178_20475) | _best | 8.906 | [1555486458178_25527](https://philly/#/job/eu2/ipgsrch/1555486458178_25527) | 21.85 | 3.51 | 13.98 |
| 0.0 | 0.2 | 0.0 | 0.0001 | 0.0 | [1555486458178_20476](https://philly/#/job/eu2/ipgsrch/1555486458178_20476) | _best | 6.589 | [1555486458178_25528](https://philly/#/job/eu2/ipgsrch/1555486458178_25528) | 24.82 | 4.67 | 15.68 |
| 0.0 | 0.2 | 0.1 | 0.0001 | 0.0 | [1555486458178_20477](https://philly/#/job/eu2/ipgsrch/1555486458178_20477) | _best | 8.085 | [1555486458178_25529](https://philly/#/job/eu2/ipgsrch/1555486458178_25529) | 22.77 | 3.82 | 14.76 |
| 0.0 | 0.2 | 0.2 | 0.0001 | 0.0 | [1555486458178_20478](https://philly/#/job/eu2/ipgsrch/1555486458178_20478) | _best | 9.499 | [1555486458178_25530](https://philly/#/job/eu2/ipgsrch/1555486458178_25530) | 20.27 | 3.16 | 13.35 |
| 0.0 | 0.2 | 0.0 | 0.0001 | 1e-05 | [1555486458178_20479](https://philly/#/job/eu2/ipgsrch/1555486458178_20479) | _best | 6.701 | [1555486458178_25531](https://philly/#/job/eu2/ipgsrch/1555486458178_25531) | 23.83 | 4.29 | 15.21 |
| 0.0 | 0.2 | 0.1 | 0.0001 | 1e-05 | [1555486458178_20480](https://philly/#/job/eu2/ipgsrch/1555486458178_20480) | _best | 8.052 | [1555486458178_25532](https://philly/#/job/eu2/ipgsrch/1555486458178_25532) | 22.94 | 3.93 | 14.83 |
| 0.0 | 0.2 | 0.2 | 0.0001 | 1e-05 | [1555486458178_20481](https://philly/#/job/eu2/ipgsrch/1555486458178_20481) | _best | 9.196 | [1555486458178_25533](https://philly/#/job/eu2/ipgsrch/1555486458178_25533) | 20.23 | 3.02 | 13.20 |
| 0.0 | 0.2 | 0.0 | 0.0001 | 0.0001 | [1555486458178_20482](https://philly/#/job/eu2/ipgsrch/1555486458178_20482) | _best | 6.701 | [1555486458178_25534](https://philly/#/job/eu2/ipgsrch/1555486458178_25534) | 23.83 | 4.29 | 15.21 |
| 0.0 | 0.2 | 0.1 | 0.0001 | 0.0001 | [1555486458178_20483](https://philly/#/job/eu2/ipgsrch/1555486458178_20483) | _best | 8.04 | [1555486458178_25535](https://philly/#/job/eu2/ipgsrch/1555486458178_25535) | 22.93 | 3.94 | 14.78 |
| 0.0 | 0.2 | 0.2 | 0.0001 | 0.0001 | [1555486458178_20484](https://philly/#/job/eu2/ipgsrch/1555486458178_20484) | _best | 9.196 | [1555486458178_25536](https://philly/#/job/eu2/ipgsrch/1555486458178_25536) | 20.23 | 3.02 | 13.20 |
| 0.1 | 0.2 | 0.0 | 0.0001 | 0.0 | [1555486458178_20485](https://philly/#/job/eu2/ipgsrch/1555486458178_20485) | _best | 7.323 | [1555486458178_25537](https://philly/#/job/eu2/ipgsrch/1555486458178_25537) | 20.21 | 3.24 | 13.50 |
| 0.1 | 0.2 | 0.1 | 0.0001 | 0.0 | [1555486458178_20486](https://philly/#/job/eu2/ipgsrch/1555486458178_20486) | _best | 8.785 | [1555486458178_25538](https://philly/#/job/eu2/ipgsrch/1555486458178_25538) | 20.36 | 3.27 | 13.31 |
| 0.1 | 0.2 | 0.2 | 0.0001 | 0.0 | [1555486458178_20487](https://philly/#/job/eu2/ipgsrch/1555486458178_20487) | _best | 9.233 | [1555486458178_25539](https://philly/#/job/eu2/ipgsrch/1555486458178_25539) | 19.09 | 2.89 | 12.92 |
| 0.1 | 0.2 | 0.0 | 0.0001 | 1e-05 | [1555486458178_20488](https://philly/#/job/eu2/ipgsrch/1555486458178_20488) | _best | 7.323 | [1555486458178_25540](https://philly/#/job/eu2/ipgsrch/1555486458178_25540) | 20.21 | 3.24 | 13.50 |
| 0.1 | 0.2 | 0.1 | 0.0001 | 1e-05 | [1555486458178_20489](https://philly/#/job/eu2/ipgsrch/1555486458178_20489) | _best | 8.374 | [1555486458178_25541](https://philly/#/job/eu2/ipgsrch/1555486458178_25541) | 20.03 | 3.13 | 13.26 |
| 0.1 | 0.2 | 0.2 | 0.0001 | 1e-05 | [1555486458178_20490](https://philly/#/job/eu2/ipgsrch/1555486458178_20490) | _best | 9.233 | [1555486458178_25542](https://philly/#/job/eu2/ipgsrch/1555486458178_25542) | 19.09 | 2.89 | 12.92 |
| 0.1 | 0.2 | 0.0 | 0.0001 | 0.0001 | [1555486458178_20491](https://philly/#/job/eu2/ipgsrch/1555486458178_20491) | _best | 7.323 | [1555486458178_25543](https://philly/#/job/eu2/ipgsrch/1555486458178_25543) | 20.21 | 3.24 | 13.50 |
| 0.1 | 0.2 | 0.1 | 0.0001 | 0.0001 | [1555486458178_20492](https://philly/#/job/eu2/ipgsrch/1555486458178_20492) | _best | 8.374 | [1555486458178_25544](https://philly/#/job/eu2/ipgsrch/1555486458178_25544) | 20.03 | 3.13 | 13.26 |
| 0.1 | 0.2 | 0.2 | 0.0001 | 0.0001 | [1555486458178_20493](https://philly/#/job/eu2/ipgsrch/1555486458178_20493) | _best | 9.233 | [1555486458178_25545](https://philly/#/job/eu2/ipgsrch/1555486458178_25545) | 19.09 | 2.89 | 12.92 |
| 0.2 | 0.2 | 0.0 | 0.0001 | 0.0 | [1555486458178_20494](https://philly/#/job/eu2/ipgsrch/1555486458178_20494) | _best | 6.847 | [1555486458178_25546](https://philly/#/job/eu2/ipgsrch/1555486458178_25546) | 23.62 | 4.28 | 15.20 |
| 0.2 | 0.2 | 0.1 | 0.0001 | 0.0 | [1555486458178_20495](https://philly/#/job/eu2/ipgsrch/1555486458178_20495) | _best | 7.803 | [1555486458178_25547](https://philly/#/job/eu2/ipgsrch/1555486458178_25547) | 25.20 | 4.87 | 16.03 |
| 0.2 | 0.2 | 0.2 | 0.0001 | 0.0 | [1555486458178_20496](https://philly/#/job/eu2/ipgsrch/1555486458178_20496) | _best | 9.178 | [1555486458178_25548](https://philly/#/job/eu2/ipgsrch/1555486458178_25548) | 22.02 | 3.65 | 14.27 |
| 0.2 | 0.2 | 0.0 | 0.0001 | 1e-05 | [1555486458178_20497](https://philly/#/job/eu2/ipgsrch/1555486458178_20497) | _best | 6.883 | [1555486458178_25549](https://philly/#/job/eu2/ipgsrch/1555486458178_25549) | 23.18 | 4.05 | 14.99 |
| 0.2 | 0.2 | 0.1 | 0.0001 | 1e-05 | [1555486458178_23263](https://philly/#/job/eu2/ipgsrch/1555486458178_23263) | _best | 8.527 | [1555486458178_25550](https://philly/#/job/eu2/ipgsrch/1555486458178_25550) | 20.58 | 3.23 | 13.50 |
| 0.2 | 0.2 | 0.2 | 0.0001 | 1e-05 | [1555486458178_20499](https://philly/#/job/eu2/ipgsrch/1555486458178_20499) | _best | 9.16 | [1555486458178_25551](https://philly/#/job/eu2/ipgsrch/1555486458178_25551) | --- | --- | --- |
| 0.2 | 0.2 | 0.0 | 0.0001 | 0.0001 | [1555486458178_20500](https://philly/#/job/eu2/ipgsrch/1555486458178_20500) | _best | 6.883 | [1555486458178_25552](https://philly/#/job/eu2/ipgsrch/1555486458178_25552) | --- | --- | --- |
| 0.2 | 0.2 | 0.1 | 0.0001 | 0.0001 | [1555486458178_20501](https://philly/#/job/eu2/ipgsrch/1555486458178_20501) | _best | 7.803 | [1555486458178_25553](https://philly/#/job/eu2/ipgsrch/1555486458178_25553) | --- | --- | --- |
| 0.2 | 0.2 | 0.2 | 0.0001 | 0.0001 | [1555486458178_20502](https://philly/#/job/eu2/ipgsrch/1555486458178_20502) | _best | 9.178 | [1555486458178_25554](https://philly/#/job/eu2/ipgsrch/1555486458178_25554) | --- | --- | --- |
| 0.0 | 0.3 | 0.0 | 0.0001 | 0.0 | [1555486458178_20503](https://philly/#/job/eu2/ipgsrch/1555486458178_20503) | _best | 7.734 | [1555486458178_25555](https://philly/#/job/eu2/ipgsrch/1555486458178_25555) | --- | --- | --- |
| 0.0 | 0.3 | 0.1 | 0.0001 | 0.0 | [1555486458178_20504](https://philly/#/job/eu2/ipgsrch/1555486458178_20504) | _best | 8.801 | [1555486458178_25556](https://philly/#/job/eu2/ipgsrch/1555486458178_25556) | --- | --- | --- |
| 0.0 | 0.3 | 0.2 | 0.0001 | 0.0 | [1555486458178_20505](https://philly/#/job/eu2/ipgsrch/1555486458178_20505) | _best | 9.512 | [1555486458178_25557](https://philly/#/job/eu2/ipgsrch/1555486458178_25557) | --- | --- | --- |
| 0.0 | 0.3 | 0.0 | 0.0001 | 1e-05 | [1555486458178_20506](https://philly/#/job/eu2/ipgsrch/1555486458178_20506) | _best | 7.131 | [1555486458178_25558](https://philly/#/job/eu2/ipgsrch/1555486458178_25558) | --- | --- | --- |
| 0.0 | 0.3 | 0.1 | 0.0001 | 1e-05 | [1555486458178_20507](https://philly/#/job/eu2/ipgsrch/1555486458178_20507) | _best | 8.255 | [1555486458178_25559](https://philly/#/job/eu2/ipgsrch/1555486458178_25559) | --- | --- | --- |
| 0.0 | 0.3 | 0.2 | 0.0001 | 1e-05 | [1555486458178_20508](https://philly/#/job/eu2/ipgsrch/1555486458178_20508) | _best | 9.512 | [1555486458178_25560](https://philly/#/job/eu2/ipgsrch/1555486458178_25560) | --- | --- | --- |
| 0.0 | 0.3 | 0.0 | 0.0001 | 0.0001 | [1555486458178_20509](https://philly/#/job/eu2/ipgsrch/1555486458178_20509) | _best | 7.215 | [1555486458178_25561](https://philly/#/job/eu2/ipgsrch/1555486458178_25561) | --- | --- | --- |
| 0.0 | 0.3 | 0.1 | 0.0001 | 0.0001 | [1555486458178_20510](https://philly/#/job/eu2/ipgsrch/1555486458178_20510) | _best | 8.687 | [1555486458178_25562](https://philly/#/job/eu2/ipgsrch/1555486458178_25562) | --- | --- | --- |
| 0.0 | 0.3 | 0.2 | 0.0001 | 0.0001 | [1555486458178_20511](https://philly/#/job/eu2/ipgsrch/1555486458178_20511) | _best | 9.512 | [1555486458178_25563](https://philly/#/job/eu2/ipgsrch/1555486458178_25563) | --- | --- | --- |
| 0.1 | 0.3 | 0.0 | 0.0001 | 0.0 | [1555486458178_20512](https://philly/#/job/eu2/ipgsrch/1555486458178_20512) | _best | 8.093 | [1555486458178_25564](https://philly/#/job/eu2/ipgsrch/1555486458178_25564) | --- | --- | --- |
| 0.1 | 0.3 | 0.1 | 0.0001 | 0.0 | [1555486458178_20513](https://philly/#/job/eu2/ipgsrch/1555486458178_20513) | _best | 8.706 | [1555486458178_25565](https://philly/#/job/eu2/ipgsrch/1555486458178_25565) | --- | --- | --- |
| 0.1 | 0.3 | 0.2 | 0.0001 | 0.0 | [1555486458178_20514](https://philly/#/job/eu2/ipgsrch/1555486458178_20514) | _best | 9.275 | [1555486458178_25566](https://philly/#/job/eu2/ipgsrch/1555486458178_25566) | --- | --- | --- |
| 0.1 | 0.3 | 0.0 | 0.0001 | 1e-05 | [1555486458178_25165](https://philly/#/job/eu2/ipgsrch/1555486458178_25165) | _best | 1000.0 | [1555486458178_25567](https://philly/#/job/eu2/ipgsrch/1555486458178_25567) | --- | --- | --- |
| 0.1 | 0.3 | 0.1 | 0.0001 | 1e-05 | [1555486458178_23264](https://philly/#/job/eu2/ipgsrch/1555486458178_23264) | _best | 8.706 | [1555486458178_25569](https://philly/#/job/eu2/ipgsrch/1555486458178_25569) | --- | --- | --- |
| 0.1 | 0.3 | 0.2 | 0.0001 | 1e-05 | [1555486458178_20517](https://philly/#/job/eu2/ipgsrch/1555486458178_20517) | _best | 9.218 | [1555486458178_25570](https://philly/#/job/eu2/ipgsrch/1555486458178_25570) | --- | --- | --- |
| 0.1 | 0.3 | 0.0 | 0.0001 | 0.0001 | [1555486458178_20518](https://philly/#/job/eu2/ipgsrch/1555486458178_20518) | _best | 7.116 | [1555486458178_25571](https://philly/#/job/eu2/ipgsrch/1555486458178_25571) | --- | --- | --- |
| 0.1 | 0.3 | 0.1 | 0.0001 | 0.0001 | [1555486458178_20519](https://philly/#/job/eu2/ipgsrch/1555486458178_20519) | _best | 7.926 | [1555486458178_25572](https://philly/#/job/eu2/ipgsrch/1555486458178_25572) | --- | --- | --- |
| 0.1 | 0.3 | 0.2 | 0.0001 | 0.0001 | [1555486458178_20520](https://philly/#/job/eu2/ipgsrch/1555486458178_20520) | _best | 9.649 | [1555486458178_25573](https://philly/#/job/eu2/ipgsrch/1555486458178_25573) | --- | --- | --- |
| 0.2 | 0.3 | 0.0 | 0.0001 | 0.0 | [1555486458178_20521](https://philly/#/job/eu2/ipgsrch/1555486458178_20521) | _best | 7.885 | [1555486458178_25574](https://philly/#/job/eu2/ipgsrch/1555486458178_25574) | --- | --- | --- |
| 0.2 | 0.3 | 0.1 | 0.0001 | 0.0 | [1555486458178_20522](https://philly/#/job/eu2/ipgsrch/1555486458178_20522) | _best | 8.858 | [1555486458178_25575](https://philly/#/job/eu2/ipgsrch/1555486458178_25575) | --- | --- | --- |
| 0.2 | 0.3 | 0.2 | 0.0001 | 0.0 | [1555486458178_20523](https://philly/#/job/eu2/ipgsrch/1555486458178_20523) | _best | 9.781 | [1555486458178_25576](https://philly/#/job/eu2/ipgsrch/1555486458178_25576) | --- | --- | --- |
| 0.2 | 0.3 | 0.0 | 0.0001 | 1e-05 | [1555486458178_20524](https://philly/#/job/eu2/ipgsrch/1555486458178_20524) | _best | 7.885 | [1555486458178_25577](https://philly/#/job/eu2/ipgsrch/1555486458178_25577) | --- | --- | --- |
| 0.2 | 0.3 | 0.1 | 0.0001 | 1e-05 | [1555486458178_20525](https://philly/#/job/eu2/ipgsrch/1555486458178_20525) | _best | 8.949 | [1555486458178_25578](https://philly/#/job/eu2/ipgsrch/1555486458178_25578) | --- | --- | --- |
| 0.2 | 0.3 | 0.2 | 0.0001 | 1e-05 | [1555486458178_20526](https://philly/#/job/eu2/ipgsrch/1555486458178_20526) | _best | 9.781 | [1555486458178_25579](https://philly/#/job/eu2/ipgsrch/1555486458178_25579) | --- | --- | --- |
| 0.2 | 0.3 | 0.0 | 0.0001 | 0.0001 | [1555486458178_20527](https://philly/#/job/eu2/ipgsrch/1555486458178_20527) | _best | 7.885 | [1555486458178_25580](https://philly/#/job/eu2/ipgsrch/1555486458178_25580) | --- | --- | --- |
| 0.2 | 0.3 | 0.1 | 0.0001 | 0.0001 | [1555486458178_20528](https://philly/#/job/eu2/ipgsrch/1555486458178_20528) | _best | 8.858 | [1555486458178_25581](https://philly/#/job/eu2/ipgsrch/1555486458178_25581) | --- | --- | --- |
| 0.2 | 0.3 | 0.2 | 0.0001 | 0.0001 | [1555486458178_23265](https://philly/#/job/eu2/ipgsrch/1555486458178_23265) | _best | 9.815 | [1555486458178_25582](https://philly/#/job/eu2/ipgsrch/1555486458178_25582) | --- | --- | --- |
| 0.0 | 0.1 | 0.0 | 0.0002 | 0.0 | [1555486458178_18349](https://philly/#/job/eu2/ipgsrch/1555486458178_18349) | _best | 8.019 | [1555486458178_25583](https://philly/#/job/eu2/ipgsrch/1555486458178_25583) | --- | --- | --- |
| 0.0 | 0.1 | 0.1 | 0.0002 | 0.0 | [1555486458178_18350](https://philly/#/job/eu2/ipgsrch/1555486458178_18350) | _best | 8.303 | [1555486458178_25584](https://philly/#/job/eu2/ipgsrch/1555486458178_25584) | --- | --- | --- |
| 0.0 | 0.1 | 0.2 | 0.0002 | 0.0 | [1555486458178_18351](https://philly/#/job/eu2/ipgsrch/1555486458178_18351) | _best | 9.262 | [1555486458178_25585](https://philly/#/job/eu2/ipgsrch/1555486458178_25585) | --- | --- | --- |
| 0.0 | 0.1 | 0.0 | 0.0002 | 1e-05 | [1555486458178_18352](https://philly/#/job/eu2/ipgsrch/1555486458178_18352) | _best | 7.519 | [1555486458178_25586](https://philly/#/job/eu2/ipgsrch/1555486458178_25586) | --- | --- | --- |
| 0.0 | 0.1 | 0.1 | 0.0002 | 1e-05 | [1555486458178_18353](https://philly/#/job/eu2/ipgsrch/1555486458178_18353) | _best | 8.302 | [1555486458178_25587](https://philly/#/job/eu2/ipgsrch/1555486458178_25587) | --- | --- | --- |
| 0.0 | 0.1 | 0.2 | 0.0002 | 1e-05 | [1555486458178_18354](https://philly/#/job/eu2/ipgsrch/1555486458178_18354) | _best | 9.262 | [1555486458178_25588](https://philly/#/job/eu2/ipgsrch/1555486458178_25588) | --- | --- | --- |
| 0.0 | 0.1 | 0.0 | 0.0002 | 0.0001 | [1555486458178_18355](https://philly/#/job/eu2/ipgsrch/1555486458178_18355) | _best | 7.105 | [1555486458178_25589](https://philly/#/job/eu2/ipgsrch/1555486458178_25589) | --- | --- | --- |
| 0.0 | 0.1 | 0.1 | 0.0002 | 0.0001 | [1555486458178_18356](https://philly/#/job/eu2/ipgsrch/1555486458178_18356) | _best | 8.718 | [1555486458178_25590](https://philly/#/job/eu2/ipgsrch/1555486458178_25590) | --- | --- | --- |
| 0.0 | 0.1 | 0.2 | 0.0002 | 0.0001 | [1555486458178_18357](https://philly/#/job/eu2/ipgsrch/1555486458178_18357) | _best | 9.262 | [1555486458178_25591](https://philly/#/job/eu2/ipgsrch/1555486458178_25591) | --- | --- | --- |
| 0.1 | 0.1 | 0.0 | 0.0002 | 0.0 | [1555486458178_18358](https://philly/#/job/eu2/ipgsrch/1555486458178_18358) | _best | 7.32 | [1555486458178_25592](https://philly/#/job/eu2/ipgsrch/1555486458178_25592) | --- | --- | --- |
| 0.1 | 0.1 | 0.1 | 0.0002 | 0.0 | [1555486458178_18359](https://philly/#/job/eu2/ipgsrch/1555486458178_18359) | _best | 8.238 | [1555486458178_25593](https://philly/#/job/eu2/ipgsrch/1555486458178_25593) | --- | --- | --- |
| 0.1 | 0.1 | 0.2 | 0.0002 | 0.0 | [1555486458178_18360](https://philly/#/job/eu2/ipgsrch/1555486458178_18360) | _best | 9.283 | [1555486458178_25594](https://philly/#/job/eu2/ipgsrch/1555486458178_25594) | --- | --- | --- |
| 0.1 | 0.1 | 0.0 | 0.0002 | 1e-05 | [1555486458178_18361](https://philly/#/job/eu2/ipgsrch/1555486458178_18361) | _best | 7.122 | [1555486458178_25595](https://philly/#/job/eu2/ipgsrch/1555486458178_25595) | --- | --- | --- |
| 0.1 | 0.1 | 0.1 | 0.0002 | 1e-05 | [1555486458178_18115](https://philly/#/job/eu2/ipgsrch/1555486458178_18115) | _best | 8.238 | [1555486458178_25596](https://philly/#/job/eu2/ipgsrch/1555486458178_25596) | --- | --- | --- |
| 0.1 | 0.1 | 0.2 | 0.0002 | 1e-05 | [1555486458178_18116](https://philly/#/job/eu2/ipgsrch/1555486458178_18116) | _best | 9.283 | [1555486458178_25597](https://philly/#/job/eu2/ipgsrch/1555486458178_25597) | --- | --- | --- |
| 0.1 | 0.1 | 0.0 | 0.0002 | 0.0001 | [1555486458178_18364](https://philly/#/job/eu2/ipgsrch/1555486458178_18364) | _best | 7.122 | [1555486458178_25598](https://philly/#/job/eu2/ipgsrch/1555486458178_25598) | --- | --- | --- |
| 0.1 | 0.1 | 0.1 | 0.0002 | 0.0001 | [1555486458178_18118](https://philly/#/job/eu2/ipgsrch/1555486458178_18118) | _best | 8.238 | [1555486458178_25599](https://philly/#/job/eu2/ipgsrch/1555486458178_25599) | --- | --- | --- |
| 0.1 | 0.1 | 0.2 | 0.0002 | 0.0001 | [1555486458178_18119](https://philly/#/job/eu2/ipgsrch/1555486458178_18119) | _best | 9.535 | [1555486458178_25600](https://philly/#/job/eu2/ipgsrch/1555486458178_25600) | --- | --- | --- |
| 0.2 | 0.1 | 0.0 | 0.0002 | 0.0 | [1555486458178_18367](https://philly/#/job/eu2/ipgsrch/1555486458178_18367) | _best | 7.234 | [1555486458178_25601](https://philly/#/job/eu2/ipgsrch/1555486458178_25601) | --- | --- | --- |
| 0.2 | 0.1 | 0.1 | 0.0002 | 0.0 | [1555486458178_18368](https://philly/#/job/eu2/ipgsrch/1555486458178_18368) | _best | 7.827 | [1555486458178_25602](https://philly/#/job/eu2/ipgsrch/1555486458178_25602) | --- | --- | --- |
| 0.2 | 0.1 | 0.2 | 0.0002 | 0.0 | [1555486458178_18369](https://philly/#/job/eu2/ipgsrch/1555486458178_18369) | _best | 9.432 | [1555486458178_25603](https://philly/#/job/eu2/ipgsrch/1555486458178_25603) | --- | --- | --- |
| 0.2 | 0.1 | 0.0 | 0.0002 | 1e-05 | [1555486458178_18370](https://philly/#/job/eu2/ipgsrch/1555486458178_18370) | _best | 7.922 | [1555486458178_25604](https://philly/#/job/eu2/ipgsrch/1555486458178_25604) | --- | --- | --- |
| 0.2 | 0.1 | 0.1 | 0.0002 | 1e-05 | [1555486458178_18124](https://philly/#/job/eu2/ipgsrch/1555486458178_18124) | _best | 7.776 | [1555486458178_25605](https://philly/#/job/eu2/ipgsrch/1555486458178_25605) | --- | --- | --- |
| 0.2 | 0.1 | 0.2 | 0.0002 | 1e-05 | [1555486458178_23266](https://philly/#/job/eu2/ipgsrch/1555486458178_23266) | _best | 9.541 | [1555486458178_25606](https://philly/#/job/eu2/ipgsrch/1555486458178_25606) | --- | --- | --- |
| 0.2 | 0.1 | 0.0 | 0.0002 | 0.0001 | [1555486458178_18373](https://philly/#/job/eu2/ipgsrch/1555486458178_18373) | _best | 7.234 | [1555486458178_25607](https://philly/#/job/eu2/ipgsrch/1555486458178_25607) | --- | --- | --- |
| 0.2 | 0.1 | 0.1 | 0.0002 | 0.0001 | [1555486458178_18127](https://philly/#/job/eu2/ipgsrch/1555486458178_18127) | _best | 7.776 | [1555486458178_25608](https://philly/#/job/eu2/ipgsrch/1555486458178_25608) | --- | --- | --- |
| 0.2 | 0.1 | 0.2 | 0.0002 | 0.0001 | [1555486458178_18128](https://philly/#/job/eu2/ipgsrch/1555486458178_18128) | _best | 9.246 | [1555486458178_25609](https://philly/#/job/eu2/ipgsrch/1555486458178_25609) | --- | --- | --- |
| 0.0 | 0.2 | 0.0 | 0.0002 | 0.0 | [1555486458178_18376](https://philly/#/job/eu2/ipgsrch/1555486458178_18376) | _best | 7.716 | [1555486458178_25610](https://philly/#/job/eu2/ipgsrch/1555486458178_25610) | --- | --- | --- |
| 0.0 | 0.2 | 0.1 | 0.0002 | 0.0 | [1555486458178_18377](https://philly/#/job/eu2/ipgsrch/1555486458178_18377) | _best | 8.673 | [1555486458178_25611](https://philly/#/job/eu2/ipgsrch/1555486458178_25611) | --- | --- | --- |
| 0.0 | 0.2 | 0.2 | 0.0002 | 0.0 | [1555486458178_18378](https://philly/#/job/eu2/ipgsrch/1555486458178_18378) | _best | 9.61 | [1555486458178_25612](https://philly/#/job/eu2/ipgsrch/1555486458178_25612) | --- | --- | --- |
| 0.0 | 0.2 | 0.0 | 0.0002 | 1e-05 | [1555486458178_18379](https://philly/#/job/eu2/ipgsrch/1555486458178_18379) | _best | 7.716 | [1555486458178_25613](https://philly/#/job/eu2/ipgsrch/1555486458178_25613) | --- | --- | --- |
| 0.0 | 0.2 | 0.1 | 0.0002 | 1e-05 | [1555486458178_18380](https://philly/#/job/eu2/ipgsrch/1555486458178_18380) | _best | 9.263 | [1555486458178_25614](https://philly/#/job/eu2/ipgsrch/1555486458178_25614) | --- | --- | --- |
| 0.0 | 0.2 | 0.2 | 0.0002 | 1e-05 | [1555486458178_18381](https://philly/#/job/eu2/ipgsrch/1555486458178_18381) | _best | 9.61 | [1555486458178_25615](https://philly/#/job/eu2/ipgsrch/1555486458178_25615) | --- | --- | --- |
| 0.0 | 0.2 | 0.0 | 0.0002 | 0.0001 | [1555486458178_18382](https://philly/#/job/eu2/ipgsrch/1555486458178_18382) | _best | 7.716 | [1555486458178_25616](https://philly/#/job/eu2/ipgsrch/1555486458178_25616) | --- | --- | --- |
| 0.0 | 0.2 | 0.1 | 0.0002 | 0.0001 | [1555486458178_18383](https://philly/#/job/eu2/ipgsrch/1555486458178_18383) | _best | 8.673 | [1555486458178_25617](https://philly/#/job/eu2/ipgsrch/1555486458178_25617) | --- | --- | --- |
| 0.0 | 0.2 | 0.2 | 0.0002 | 0.0001 | [1555486458178_18384](https://philly/#/job/eu2/ipgsrch/1555486458178_18384) | _best | 9.61 | [1555486458178_25618](https://philly/#/job/eu2/ipgsrch/1555486458178_25618) | --- | --- | --- |
| 0.1 | 0.2 | 0.0 | 0.0002 | 0.0 | [1555486458178_18385](https://philly/#/job/eu2/ipgsrch/1555486458178_18385) | _best | 7.497 | [1555486458178_25619](https://philly/#/job/eu2/ipgsrch/1555486458178_25619) | --- | --- | --- |
| 0.1 | 0.2 | 0.1 | 0.0002 | 0.0 | [1555486458178_18386](https://philly/#/job/eu2/ipgsrch/1555486458178_18386) | _best | 8.738 | [1555486458178_25620](https://philly/#/job/eu2/ipgsrch/1555486458178_25620) | --- | --- | --- |
| 0.1 | 0.2 | 0.2 | 0.0002 | 0.0 | [1555486458178_18387](https://philly/#/job/eu2/ipgsrch/1555486458178_18387) | _best | 9.614 | [1555486458178_25621](https://philly/#/job/eu2/ipgsrch/1555486458178_25621) | --- | --- | --- |
| 0.1 | 0.2 | 0.0 | 0.0002 | 1e-05 | [1555486458178_18388](https://philly/#/job/eu2/ipgsrch/1555486458178_18388) | _best | 7.497 | [1555486458178_25622](https://philly/#/job/eu2/ipgsrch/1555486458178_25622) | --- | --- | --- |
| 0.1 | 0.2 | 0.1 | 0.0002 | 1e-05 | [1555486458178_18142](https://philly/#/job/eu2/ipgsrch/1555486458178_18142) | _best | 8.774 | [1555486458178_25623](https://philly/#/job/eu2/ipgsrch/1555486458178_25623) | --- | --- | --- |
| 0.1 | 0.2 | 0.2 | 0.0002 | 1e-05 | [1555486458178_18143](https://philly/#/job/eu2/ipgsrch/1555486458178_18143) | _best | 9.614 | [1555486458178_25624](https://philly/#/job/eu2/ipgsrch/1555486458178_25624) | --- | --- | --- |
| 0.1 | 0.2 | 0.0 | 0.0002 | 0.0001 | [1555486458178_18391](https://philly/#/job/eu2/ipgsrch/1555486458178_18391) | _best | 7.497 | [1555486458178_25625](https://philly/#/job/eu2/ipgsrch/1555486458178_25625) | --- | --- | --- |
| 0.1 | 0.2 | 0.1 | 0.0002 | 0.0001 | [1555486458178_18145](https://philly/#/job/eu2/ipgsrch/1555486458178_18145) | _best | 8.738 | [1555486458178_25626](https://philly/#/job/eu2/ipgsrch/1555486458178_25626) | --- | --- | --- |
| 0.1 | 0.2 | 0.2 | 0.0002 | 0.0001 | [1555486458178_18146](https://philly/#/job/eu2/ipgsrch/1555486458178_18146) | _best | 9.711 | [1555486458178_25627](https://philly/#/job/eu2/ipgsrch/1555486458178_25627) | --- | --- | --- |
| 0.2 | 0.2 | 0.0 | 0.0002 | 0.0 | [1555486458178_18394](https://philly/#/job/eu2/ipgsrch/1555486458178_18394) | _best | 8.334 | [1555486458178_25628](https://philly/#/job/eu2/ipgsrch/1555486458178_25628) | --- | --- | --- |
| 0.2 | 0.2 | 0.1 | 0.0002 | 0.0 | [1555486458178_18395](https://philly/#/job/eu2/ipgsrch/1555486458178_18395) | _best | 8.582 | [1555486458178_25629](https://philly/#/job/eu2/ipgsrch/1555486458178_25629) | --- | --- | --- |
| 0.2 | 0.2 | 0.2 | 0.0002 | 0.0 | [1555486458178_18396](https://philly/#/job/eu2/ipgsrch/1555486458178_18396) | _best | 9.687 | [1555486458178_25630](https://philly/#/job/eu2/ipgsrch/1555486458178_25630) | --- | --- | --- |
| 0.2 | 0.2 | 0.0 | 0.0002 | 1e-05 | [1555486458178_18397](https://philly/#/job/eu2/ipgsrch/1555486458178_18397) | _best | 7.698 | [1555486458178_25631](https://philly/#/job/eu2/ipgsrch/1555486458178_25631) | --- | --- | --- |
| 0.2 | 0.2 | 0.1 | 0.0002 | 1e-05 | [1555486458178_18151](https://philly/#/job/eu2/ipgsrch/1555486458178_18151) | _best | 8.582 | [1555486458178_25632](https://philly/#/job/eu2/ipgsrch/1555486458178_25632) | --- | --- | --- |
| 0.2 | 0.2 | 0.2 | 0.0002 | 1e-05 | [1555486458178_18152](https://philly/#/job/eu2/ipgsrch/1555486458178_18152) | _best | 9.687 | [1555486458178_25633](https://philly/#/job/eu2/ipgsrch/1555486458178_25633) | --- | --- | --- |
| 0.2 | 0.2 | 0.0 | 0.0002 | 0.0001 | [1555486458178_18400](https://philly/#/job/eu2/ipgsrch/1555486458178_18400) | _best | 7.698 | [1555486458178_25634](https://philly/#/job/eu2/ipgsrch/1555486458178_25634) | --- | --- | --- |
| 0.2 | 0.2 | 0.1 | 0.0002 | 0.0001 | [1555486458178_18154](https://philly/#/job/eu2/ipgsrch/1555486458178_18154) | _best | 8.847 | [1555486458178_25635](https://philly/#/job/eu2/ipgsrch/1555486458178_25635) | --- | --- | --- |
| 0.2 | 0.2 | 0.2 | 0.0002 | 0.0001 | [1555486458178_18155](https://philly/#/job/eu2/ipgsrch/1555486458178_18155) | _best | 9.788 | [1555486458178_25636](https://philly/#/job/eu2/ipgsrch/1555486458178_25636) | --- | --- | --- |
| 0.0 | 0.3 | 0.0 | 0.0002 | 0.0 | [1555486458178_18403](https://philly/#/job/eu2/ipgsrch/1555486458178_18403) | _best | 8.254 | [1555486458178_25637](https://philly/#/job/eu2/ipgsrch/1555486458178_25637) | --- | --- | --- |
| 0.0 | 0.3 | 0.1 | 0.0002 | 0.0 | [1555486458178_25166](https://philly/#/job/eu2/ipgsrch/1555486458178_25166) | _best | 1000.0 | [1555486458178_25638](https://philly/#/job/eu2/ipgsrch/1555486458178_25638) | --- | --- | --- |
| 0.0 | 0.3 | 0.2 | 0.0002 | 0.0 | [1555486458178_18405](https://philly/#/job/eu2/ipgsrch/1555486458178_18405) | _best | 10.417 | [1555486458178_25639](https://philly/#/job/eu2/ipgsrch/1555486458178_25639) | --- | --- | --- |
| 0.0 | 0.3 | 0.0 | 0.0002 | 1e-05 | [1555486458178_18406](https://philly/#/job/eu2/ipgsrch/1555486458178_18406) | _best | 8.254 | [1555486458178_25640](https://philly/#/job/eu2/ipgsrch/1555486458178_25640) | --- | --- | --- |
| 0.0 | 0.3 | 0.1 | 0.0002 | 1e-05 | [1555486458178_18407](https://philly/#/job/eu2/ipgsrch/1555486458178_18407) | _best | 9.3 | [1555486458178_25641](https://philly/#/job/eu2/ipgsrch/1555486458178_25641) | --- | --- | --- |
| 0.0 | 0.3 | 0.2 | 0.0002 | 1e-05 | [1555486458178_18408](https://philly/#/job/eu2/ipgsrch/1555486458178_18408) | _best | 10.11 | [1555486458178_25642](https://philly/#/job/eu2/ipgsrch/1555486458178_25642) | --- | --- | --- |
| 0.0 | 0.3 | 0.0 | 0.0002 | 0.0001 | [1555486458178_18409](https://philly/#/job/eu2/ipgsrch/1555486458178_18409) | _best | 8.254 | [1555486458178_25643](https://philly/#/job/eu2/ipgsrch/1555486458178_25643) | --- | --- | --- |
| 0.0 | 0.3 | 0.1 | 0.0002 | 0.0001 | [1555486458178_18410](https://philly/#/job/eu2/ipgsrch/1555486458178_18410) | _best | 9.3 | [1555486458178_25644](https://philly/#/job/eu2/ipgsrch/1555486458178_25644) | --- | --- | --- |
| 0.0 | 0.3 | 0.2 | 0.0002 | 0.0001 | [1555486458178_18411](https://philly/#/job/eu2/ipgsrch/1555486458178_18411) | _best | 10.11 | [1555486458178_25645](https://philly/#/job/eu2/ipgsrch/1555486458178_25645) | --- | --- | --- |
| 0.1 | 0.3 | 0.0 | 0.0002 | 0.0 | [1555486458178_18412](https://philly/#/job/eu2/ipgsrch/1555486458178_18412) | _best | 8.519 | [1555486458178_25646](https://philly/#/job/eu2/ipgsrch/1555486458178_25646) | --- | --- | --- |
| 0.1 | 0.3 | 0.1 | 0.0002 | 0.0 | [1555486458178_23267](https://philly/#/job/eu2/ipgsrch/1555486458178_23267) | _best | 9.393 | [1555486458178_25647](https://philly/#/job/eu2/ipgsrch/1555486458178_25647) | --- | --- | --- |
| 0.1 | 0.3 | 0.2 | 0.0002 | 0.0 | [1555486458178_23268](https://philly/#/job/eu2/ipgsrch/1555486458178_23268) | _best | 10.178 | [1555486458178_25648](https://philly/#/job/eu2/ipgsrch/1555486458178_25648) | --- | --- | --- |
| 0.1 | 0.3 | 0.0 | 0.0002 | 1e-05 | [1555486458178_23269](https://philly/#/job/eu2/ipgsrch/1555486458178_23269) | _best | 8.569 | [1555486458178_25649](https://philly/#/job/eu2/ipgsrch/1555486458178_25649) | --- | --- | --- |
| 0.1 | 0.3 | 0.1 | 0.0002 | 1e-05 | [1555486458178_18169](https://philly/#/job/eu2/ipgsrch/1555486458178_18169) | _best | 9.281 | [1555486458178_25650](https://philly/#/job/eu2/ipgsrch/1555486458178_25650) | --- | --- | --- |
| 0.1 | 0.3 | 0.2 | 0.0002 | 1e-05 | [1555486458178_18170](https://philly/#/job/eu2/ipgsrch/1555486458178_18170) | _best | 10.178 | [1555486458178_25651](https://philly/#/job/eu2/ipgsrch/1555486458178_25651) | --- | --- | --- |
| 0.1 | 0.3 | 0.0 | 0.0002 | 0.0001 | [1555486458178_18418](https://philly/#/job/eu2/ipgsrch/1555486458178_18418) | _best | 8.519 | [1555486458178_25652](https://philly/#/job/eu2/ipgsrch/1555486458178_25652) | --- | --- | --- |
| 0.1 | 0.3 | 0.1 | 0.0002 | 0.0001 | [1555486458178_18172](https://philly/#/job/eu2/ipgsrch/1555486458178_18172) | _best | 9.393 | [1555486458178_25653](https://philly/#/job/eu2/ipgsrch/1555486458178_25653) | --- | --- | --- |
| 0.1 | 0.3 | 0.2 | 0.0002 | 0.0001 | [1555486458178_18173](https://philly/#/job/eu2/ipgsrch/1555486458178_18173) | _best | 10.178 | [1555486458178_25654](https://philly/#/job/eu2/ipgsrch/1555486458178_25654) | --- | --- | --- |
| 0.2 | 0.3 | 0.0 | 0.0002 | 0.0 | [1555486458178_18421](https://philly/#/job/eu2/ipgsrch/1555486458178_18421) | _best | 8.564 | [1555486458178_25655](https://philly/#/job/eu2/ipgsrch/1555486458178_25655) | --- | --- | --- |
| 0.2 | 0.3 | 0.1 | 0.0002 | 0.0 | [1555486458178_18422](https://philly/#/job/eu2/ipgsrch/1555486458178_18422) | _best | 9.243 | [1555486458178_25656](https://philly/#/job/eu2/ipgsrch/1555486458178_25656) | --- | --- | --- |
| 0.2 | 0.3 | 0.2 | 0.0002 | 0.0 | [1555486458178_18423](https://philly/#/job/eu2/ipgsrch/1555486458178_18423) | _best | 10.105 | [1555486458178_25657](https://philly/#/job/eu2/ipgsrch/1555486458178_25657) | --- | --- | --- |
| 0.2 | 0.3 | 0.0 | 0.0002 | 1e-05 | [1555486458178_18424](https://philly/#/job/eu2/ipgsrch/1555486458178_18424) | _best | 8.828 | [1555486458178_25658](https://philly/#/job/eu2/ipgsrch/1555486458178_25658) | --- | --- | --- |
| 0.2 | 0.3 | 0.1 | 0.0002 | 1e-05 | [1555486458178_18178](https://philly/#/job/eu2/ipgsrch/1555486458178_18178) | _best | 9.243 | [1555486458178_25659](https://philly/#/job/eu2/ipgsrch/1555486458178_25659) | --- | --- | --- |
| 0.2 | 0.3 | 0.2 | 0.0002 | 1e-05 | [1555486458178_18179](https://philly/#/job/eu2/ipgsrch/1555486458178_18179) | _best | 10.105 | [1555486458178_25660](https://philly/#/job/eu2/ipgsrch/1555486458178_25660) | --- | --- | --- |
| 0.2 | 0.3 | 0.0 | 0.0002 | 0.0001 | [1555486458178_18427](https://philly/#/job/eu2/ipgsrch/1555486458178_18427) | _best | 8.265 | [1555486458178_25661](https://philly/#/job/eu2/ipgsrch/1555486458178_25661) | --- | --- | --- |
| 0.2 | 0.3 | 0.1 | 0.0002 | 0.0001 | [1555486458178_18181](https://philly/#/job/eu2/ipgsrch/1555486458178_18181) | _best | 9.304 | [1555486458178_25662](https://philly/#/job/eu2/ipgsrch/1555486458178_25662) | --- | --- | --- |
| 0.2 | 0.3 | 0.2 | 0.0002 | 0.0001 | [1555486458178_18182](https://philly/#/job/eu2/ipgsrch/1555486458178_18182) | _best | 10.105 | [1555486458178_25663](https://philly/#/job/eu2/ipgsrch/1555486458178_25663) | --- | --- | --- |
| 0.0 | 0.1 | 0.0 | 0.0003 | 0.0 | [1555486458178_18430](https://philly/#/job/eu2/ipgsrch/1555486458178_18430) | _best | 11.086 | [1555486458178_25664](https://philly/#/job/eu2/ipgsrch/1555486458178_25664) | --- | --- | --- |
| 0.0 | 0.1 | 0.1 | 0.0003 | 0.0 | [1555486458178_18431](https://philly/#/job/eu2/ipgsrch/1555486458178_18431) | _best | 12.059 | [1555486458178_25665](https://philly/#/job/eu2/ipgsrch/1555486458178_25665) | --- | --- | --- |
| 0.0 | 0.1 | 0.2 | 0.0003 | 0.0 | [1555486458178_18432](https://philly/#/job/eu2/ipgsrch/1555486458178_18432) | _best | 12.894 | [1555486458178_25666](https://philly/#/job/eu2/ipgsrch/1555486458178_25666) | --- | --- | --- |
| 0.0 | 0.1 | 0.0 | 0.0003 | 1e-05 | [1555486458178_18433](https://philly/#/job/eu2/ipgsrch/1555486458178_18433) | _best | 11.086 | [1555486458178_25667](https://philly/#/job/eu2/ipgsrch/1555486458178_25667) | --- | --- | --- |
| 0.0 | 0.1 | 0.1 | 0.0003 | 1e-05 | [1555486458178_18434](https://philly/#/job/eu2/ipgsrch/1555486458178_18434) | _best | 11.48 | [1555486458178_25668](https://philly/#/job/eu2/ipgsrch/1555486458178_25668) | --- | --- | --- |
| 0.0 | 0.1 | 0.2 | 0.0003 | 1e-05 | [1555486458178_18435](https://philly/#/job/eu2/ipgsrch/1555486458178_18435) | _best | 11.953 | [1555486458178_25669](https://philly/#/job/eu2/ipgsrch/1555486458178_25669) | --- | --- | --- |
| 0.0 | 0.1 | 0.0 | 0.0003 | 0.0001 | [1555486458178_18436](https://philly/#/job/eu2/ipgsrch/1555486458178_18436) | _best | 11.179 | [1555486458178_25670](https://philly/#/job/eu2/ipgsrch/1555486458178_25670) | --- | --- | --- |
| 0.0 | 0.1 | 0.1 | 0.0003 | 0.0001 | [1555486458178_18437](https://philly/#/job/eu2/ipgsrch/1555486458178_18437) | _best | 11.848 | [1555486458178_25671](https://philly/#/job/eu2/ipgsrch/1555486458178_25671) | --- | --- | --- |
| 0.0 | 0.1 | 0.2 | 0.0003 | 0.0001 | [1555486458178_18438](https://philly/#/job/eu2/ipgsrch/1555486458178_18438) | _best | 12.5 | [1555486458178_25672](https://philly/#/job/eu2/ipgsrch/1555486458178_25672) | --- | --- | --- |
| 0.1 | 0.1 | 0.0 | 0.0003 | 0.0 | [1555486458178_18439](https://philly/#/job/eu2/ipgsrch/1555486458178_18439) | _best | 11.209 | [1555486458178_25673](https://philly/#/job/eu2/ipgsrch/1555486458178_25673) | --- | --- | --- |
| 0.1 | 0.1 | 0.1 | 0.0003 | 0.0 | [1555486458178_25167](https://philly/#/job/eu2/ipgsrch/1555486458178_25167) | _best | 1000.0 | [1555486458178_25674](https://philly/#/job/eu2/ipgsrch/1555486458178_25674) | --- | --- | --- |
| 0.1 | 0.1 | 0.2 | 0.0003 | 0.0 | [1555486458178_18442](https://philly/#/job/eu2/ipgsrch/1555486458178_18442) | _best | 11.322 | [1555486458178_25676](https://philly/#/job/eu2/ipgsrch/1555486458178_25676) | --- | --- | --- |
| 0.1 | 0.1 | 0.0 | 0.0003 | 1e-05 | [1555486458178_18443](https://philly/#/job/eu2/ipgsrch/1555486458178_18443) | _best | 11.038 | [1555486458178_25677](https://philly/#/job/eu2/ipgsrch/1555486458178_25677) | --- | --- | --- |
| 0.1 | 0.1 | 0.1 | 0.0003 | 1e-05 | [1555486458178_18196](https://philly/#/job/eu2/ipgsrch/1555486458178_18196) | _best | 11.46 | [1555486458178_25678](https://philly/#/job/eu2/ipgsrch/1555486458178_25678) | --- | --- | --- |
| 0.1 | 0.1 | 0.2 | 0.0003 | 1e-05 | [1555486458178_18197](https://philly/#/job/eu2/ipgsrch/1555486458178_18197) | _best | 11.315 | [1555486458178_25679](https://philly/#/job/eu2/ipgsrch/1555486458178_25679) | --- | --- | --- |
| 0.1 | 0.1 | 0.0 | 0.0003 | 0.0001 | [1555486458178_18446](https://philly/#/job/eu2/ipgsrch/1555486458178_18446) | _best | 11.192 | [1555486458178_25680](https://philly/#/job/eu2/ipgsrch/1555486458178_25680) | --- | --- | --- |
| 0.1 | 0.1 | 0.1 | 0.0003 | 0.0001 | [1555486458178_18199](https://philly/#/job/eu2/ipgsrch/1555486458178_18199) | _best | 11.278 | [1555486458178_25681](https://philly/#/job/eu2/ipgsrch/1555486458178_25681) | --- | --- | --- |
| 0.1 | 0.1 | 0.2 | 0.0003 | 0.0001 | [1555486458178_18200](https://philly/#/job/eu2/ipgsrch/1555486458178_18200) | _best | 12.015 | [1555486458178_25682](https://philly/#/job/eu2/ipgsrch/1555486458178_25682) | --- | --- | --- |
| 0.2 | 0.1 | 0.0 | 0.0003 | 0.0 | [1555486458178_18449](https://philly/#/job/eu2/ipgsrch/1555486458178_18449) | _best | 10.946 | [1555486458178_25683](https://philly/#/job/eu2/ipgsrch/1555486458178_25683) | --- | --- | --- |
| 0.2 | 0.1 | 0.1 | 0.0003 | 0.0 | [1555486458178_18450](https://philly/#/job/eu2/ipgsrch/1555486458178_18450) | _best | 11.501 | [1555486458178_25684](https://philly/#/job/eu2/ipgsrch/1555486458178_25684) | --- | --- | --- |
| 0.2 | 0.1 | 0.2 | 0.0003 | 0.0 | [1555486458178_18451](https://philly/#/job/eu2/ipgsrch/1555486458178_18451) | _best | 10.395 | [1555486458178_25685](https://philly/#/job/eu2/ipgsrch/1555486458178_25685) | --- | --- | --- |
| 0.2 | 0.1 | 0.0 | 0.0003 | 1e-05 | [1555486458178_18452](https://philly/#/job/eu2/ipgsrch/1555486458178_18452) | _best | 10.946 | [1555486458178_25686](https://philly/#/job/eu2/ipgsrch/1555486458178_25686) | --- | --- | --- |
| 0.2 | 0.1 | 0.1 | 0.0003 | 1e-05 | [1555486458178_18205](https://philly/#/job/eu2/ipgsrch/1555486458178_18205) | _best | 11.501 | [1555486458178_25687](https://philly/#/job/eu2/ipgsrch/1555486458178_25687) | --- | --- | --- |
| 0.2 | 0.1 | 0.2 | 0.0003 | 1e-05 | [1555486458178_18206](https://philly/#/job/eu2/ipgsrch/1555486458178_18206) | _best | 11.173 | [1555486458178_25688](https://philly/#/job/eu2/ipgsrch/1555486458178_25688) | --- | --- | --- |
| 0.2 | 0.1 | 0.0 | 0.0003 | 0.0001 | [1555486458178_18455](https://philly/#/job/eu2/ipgsrch/1555486458178_18455) | _best | 11.374 | [1555486458178_25689](https://philly/#/job/eu2/ipgsrch/1555486458178_25689) | --- | --- | --- |
| 0.2 | 0.1 | 0.1 | 0.0003 | 0.0001 | [1555486458178_18208](https://philly/#/job/eu2/ipgsrch/1555486458178_18208) | _best | 11.457 | [1555486458178_25690](https://philly/#/job/eu2/ipgsrch/1555486458178_25690) | --- | --- | --- |
| 0.2 | 0.1 | 0.2 | 0.0003 | 0.0001 | [1555486458178_18209](https://philly/#/job/eu2/ipgsrch/1555486458178_18209) | _best | 9.851 | [1555486458178_25691](https://philly/#/job/eu2/ipgsrch/1555486458178_25691) | --- | --- | --- |
| 0.0 | 0.2 | 0.0 | 0.0003 | 0.0 | [1555486458178_18459](https://philly/#/job/eu2/ipgsrch/1555486458178_18459) | _best | 11.516 | [1555486458178_25692](https://philly/#/job/eu2/ipgsrch/1555486458178_25692) | --- | --- | --- |
| 0.0 | 0.2 | 0.1 | 0.0003 | 0.0 | [1555486458178_18460](https://philly/#/job/eu2/ipgsrch/1555486458178_18460) | _best | 11.426 | [1555486458178_25693](https://philly/#/job/eu2/ipgsrch/1555486458178_25693) | --- | --- | --- |
| 0.0 | 0.2 | 0.2 | 0.0003 | 0.0 | [1555486458178_18461](https://philly/#/job/eu2/ipgsrch/1555486458178_18461) | _best | 12.607 | [1555486458178_25694](https://philly/#/job/eu2/ipgsrch/1555486458178_25694) | --- | --- | --- |
| 0.0 | 0.2 | 0.0 | 0.0003 | 1e-05 | [1555486458178_18462](https://philly/#/job/eu2/ipgsrch/1555486458178_18462) | _best | 11.036 | [1555486458178_25695](https://philly/#/job/eu2/ipgsrch/1555486458178_25695) | --- | --- | --- |
| 0.0 | 0.2 | 0.1 | 0.0003 | 1e-05 | [1555486458178_18463](https://philly/#/job/eu2/ipgsrch/1555486458178_18463) | _best | 11.413 | [1555486458178_25696](https://philly/#/job/eu2/ipgsrch/1555486458178_25696) | --- | --- | --- |
| 0.0 | 0.2 | 0.2 | 0.0003 | 1e-05 | [1555486458178_18464](https://philly/#/job/eu2/ipgsrch/1555486458178_18464) | _best | 12.607 | [1555486458178_25697](https://philly/#/job/eu2/ipgsrch/1555486458178_25697) | --- | --- | --- |
| 0.0 | 0.2 | 0.0 | 0.0003 | 0.0001 | [1555486458178_18465](https://philly/#/job/eu2/ipgsrch/1555486458178_18465) | _best | 11.891 | [1555486458178_25698](https://philly/#/job/eu2/ipgsrch/1555486458178_25698) | --- | --- | --- |
| 0.0 | 0.2 | 0.1 | 0.0003 | 0.0001 | [1555486458178_18466](https://philly/#/job/eu2/ipgsrch/1555486458178_18466) | _best | 11.99 | [1555486458178_25699](https://philly/#/job/eu2/ipgsrch/1555486458178_25699) | --- | --- | --- |
| 0.0 | 0.2 | 0.2 | 0.0003 | 0.0001 | [1555486458178_18467](https://philly/#/job/eu2/ipgsrch/1555486458178_18467) | _best | 11.926 | [1555486458178_25700](https://philly/#/job/eu2/ipgsrch/1555486458178_25700) | --- | --- | --- |
| 0.1 | 0.2 | 0.0 | 0.0003 | 0.0 | [1555486458178_18468](https://philly/#/job/eu2/ipgsrch/1555486458178_18468) | _best | 10.773 | [1555486458178_25701](https://philly/#/job/eu2/ipgsrch/1555486458178_25701) | --- | --- | --- |
| 0.1 | 0.2 | 0.1 | 0.0003 | 0.0 | [1555486458178_18469](https://philly/#/job/eu2/ipgsrch/1555486458178_18469) | _best | 11.625 | [1555486458178_25702](https://philly/#/job/eu2/ipgsrch/1555486458178_25702) | --- | --- | --- |
| 0.1 | 0.2 | 0.2 | 0.0003 | 0.0 | [1555486458178_18470](https://philly/#/job/eu2/ipgsrch/1555486458178_18470) | _best | 12.424 | [1555486458178_25703](https://philly/#/job/eu2/ipgsrch/1555486458178_25703) | --- | --- | --- |
| 0.1 | 0.2 | 0.0 | 0.0003 | 1e-05 | [1555486458178_18471](https://philly/#/job/eu2/ipgsrch/1555486458178_18471) | _best | 10.773 | [1555486458178_25704](https://philly/#/job/eu2/ipgsrch/1555486458178_25704) | --- | --- | --- |
| 0.1 | 0.2 | 0.1 | 0.0003 | 1e-05 | [1555486458178_18223](https://philly/#/job/eu2/ipgsrch/1555486458178_18223) | _best | 12.126 | [1555486458178_25705](https://philly/#/job/eu2/ipgsrch/1555486458178_25705) | --- | --- | --- |
| 0.1 | 0.2 | 0.2 | 0.0003 | 1e-05 | [1555486458178_18224](https://philly/#/job/eu2/ipgsrch/1555486458178_18224) | _best | 11.985 | [1555486458178_25706](https://philly/#/job/eu2/ipgsrch/1555486458178_25706) | --- | --- | --- |
| 0.1 | 0.2 | 0.0 | 0.0003 | 0.0001 | [1555486458178_18474](https://philly/#/job/eu2/ipgsrch/1555486458178_18474) | _best | 11.197 | [1555486458178_25707](https://philly/#/job/eu2/ipgsrch/1555486458178_25707) | --- | --- | --- |
| 0.1 | 0.2 | 0.1 | 0.0003 | 0.0001 | [1555486458178_18226](https://philly/#/job/eu2/ipgsrch/1555486458178_18226) | _best | 11.445 | [1555486458178_25708](https://philly/#/job/eu2/ipgsrch/1555486458178_25708) | --- | --- | --- |
| 0.1 | 0.2 | 0.2 | 0.0003 | 0.0001 | [1555486458178_18227](https://philly/#/job/eu2/ipgsrch/1555486458178_18227) | _best | 11.968 | [1555486458178_25709](https://philly/#/job/eu2/ipgsrch/1555486458178_25709) | --- | --- | --- |
| 0.2 | 0.2 | 0.0 | 0.0003 | 0.0 | [1555486458178_18477](https://philly/#/job/eu2/ipgsrch/1555486458178_18477) | _best | 11.105 | [1555486458178_25710](https://philly/#/job/eu2/ipgsrch/1555486458178_25710) | --- | --- | --- |
| 0.2 | 0.2 | 0.1 | 0.0003 | 0.0 | [1555486458178_18478](https://philly/#/job/eu2/ipgsrch/1555486458178_18478) | _best | 13.139 | [1555486458178_25711](https://philly/#/job/eu2/ipgsrch/1555486458178_25711) | --- | --- | --- |
| 0.2 | 0.2 | 0.2 | 0.0003 | 0.0 | [1555486458178_18479](https://philly/#/job/eu2/ipgsrch/1555486458178_18479) | _best | 12.114 | [1555486458178_25712](https://philly/#/job/eu2/ipgsrch/1555486458178_25712) | --- | --- | --- |
| 0.2 | 0.2 | 0.0 | 0.0003 | 1e-05 | [1555486458178_18480](https://philly/#/job/eu2/ipgsrch/1555486458178_18480) | _best | 11.105 | [1555486458178_25713](https://philly/#/job/eu2/ipgsrch/1555486458178_25713) | --- | --- | --- |
| 0.2 | 0.2 | 0.1 | 0.0003 | 1e-05 | [1555486458178_18232](https://philly/#/job/eu2/ipgsrch/1555486458178_18232) | _best | 12.503 | [1555486458178_25714](https://philly/#/job/eu2/ipgsrch/1555486458178_25714) | --- | --- | --- |
| 0.2 | 0.2 | 0.2 | 0.0003 | 1e-05 | [1555486458178_18233](https://philly/#/job/eu2/ipgsrch/1555486458178_18233) | _best | 12.794 | [1555486458178_25715](https://philly/#/job/eu2/ipgsrch/1555486458178_25715) | --- | --- | --- |
| 0.2 | 0.2 | 0.0 | 0.0003 | 0.0001 | [1555486458178_18483](https://philly/#/job/eu2/ipgsrch/1555486458178_18483) | _best | 11.255 | [1555486458178_25716](https://philly/#/job/eu2/ipgsrch/1555486458178_25716) | --- | --- | --- |
| 0.2 | 0.2 | 0.1 | 0.0003 | 0.0001 | [1555486458178_18235](https://philly/#/job/eu2/ipgsrch/1555486458178_18235) | _best | 11.481 | [1555486458178_25717](https://philly/#/job/eu2/ipgsrch/1555486458178_25717) | --- | --- | --- |
| 0.2 | 0.2 | 0.2 | 0.0003 | 0.0001 | [1555486458178_18236](https://philly/#/job/eu2/ipgsrch/1555486458178_18236) | _best | 12.889 | [1555486458178_25718](https://philly/#/job/eu2/ipgsrch/1555486458178_25718) | --- | --- | --- |
| 0.0 | 0.3 | 0.0 | 0.0003 | 0.0 | [1555486458178_18486](https://philly/#/job/eu2/ipgsrch/1555486458178_18486) | _best | 11.38 | [1555486458178_25719](https://philly/#/job/eu2/ipgsrch/1555486458178_25719) | --- | --- | --- |
| 0.0 | 0.3 | 0.1 | 0.0003 | 0.0 | [1555486458178_18488](https://philly/#/job/eu2/ipgsrch/1555486458178_18488) | _best | 11.545 | [1555486458178_25720](https://philly/#/job/eu2/ipgsrch/1555486458178_25720) | --- | --- | --- |
| 0.0 | 0.3 | 0.2 | 0.0003 | 0.0 | [1555486458178_18489](https://philly/#/job/eu2/ipgsrch/1555486458178_18489) | _best | 12.045 | [1555486458178_25721](https://philly/#/job/eu2/ipgsrch/1555486458178_25721) | --- | --- | --- |
| 0.0 | 0.3 | 0.0 | 0.0003 | 1e-05 | [1555486458178_18490](https://philly/#/job/eu2/ipgsrch/1555486458178_18490) | _best | 10.932 | [1555486458178_25722](https://philly/#/job/eu2/ipgsrch/1555486458178_25722) | --- | --- | --- |
| 0.0 | 0.3 | 0.1 | 0.0003 | 1e-05 | [1555486458178_18491](https://philly/#/job/eu2/ipgsrch/1555486458178_18491) | _best | 11.642 | [1555486458178_25723](https://philly/#/job/eu2/ipgsrch/1555486458178_25723) | --- | --- | --- |
| 0.0 | 0.3 | 0.2 | 0.0003 | 1e-05 | [1555486458178_18492](https://philly/#/job/eu2/ipgsrch/1555486458178_18492) | _best | 12.746 | [1555486458178_25724](https://philly/#/job/eu2/ipgsrch/1555486458178_25724) | --- | --- | --- |
| 0.0 | 0.3 | 0.0 | 0.0003 | 0.0001 | [1555486458178_18493](https://philly/#/job/eu2/ipgsrch/1555486458178_18493) | _best | 11.2 | [1555486458178_25725](https://philly/#/job/eu2/ipgsrch/1555486458178_25725) | --- | --- | --- |
| 0.0 | 0.3 | 0.1 | 0.0003 | 0.0001 | [1555486458178_18494](https://philly/#/job/eu2/ipgsrch/1555486458178_18494) | _best | 11.418 | [1555486458178_25726](https://philly/#/job/eu2/ipgsrch/1555486458178_25726) | --- | --- | --- |
| 0.0 | 0.3 | 0.2 | 0.0003 | 0.0001 | [1555486458178_18495](https://philly/#/job/eu2/ipgsrch/1555486458178_18495) | _best | 12.039 | [1555486458178_25727](https://philly/#/job/eu2/ipgsrch/1555486458178_25727) | --- | --- | --- |
| 0.1 | 0.3 | 0.0 | 0.0003 | 0.0 | [1555486458178_18496](https://philly/#/job/eu2/ipgsrch/1555486458178_18496) | _best | 10.866 | [1555486458178_25728](https://philly/#/job/eu2/ipgsrch/1555486458178_25728) | --- | --- | --- |
| 0.1 | 0.3 | 0.1 | 0.0003 | 0.0 | [1555486458178_18497](https://philly/#/job/eu2/ipgsrch/1555486458178_18497) | _best | 11.651 | [1555486458178_25729](https://philly/#/job/eu2/ipgsrch/1555486458178_25729) | --- | --- | --- |
| 0.1 | 0.3 | 0.2 | 0.0003 | 0.0 | [1555486458178_18498](https://philly/#/job/eu2/ipgsrch/1555486458178_18498) | _best | 12.004 | [1555486458178_25730](https://philly/#/job/eu2/ipgsrch/1555486458178_25730) | --- | --- | --- |
| 0.1 | 0.3 | 0.0 | 0.0003 | 1e-05 | [1555486458178_18499](https://philly/#/job/eu2/ipgsrch/1555486458178_18499) | _best | 11.253 | [1555486458178_25731](https://philly/#/job/eu2/ipgsrch/1555486458178_25731) | --- | --- | --- |
| 0.1 | 0.3 | 0.1 | 0.0003 | 1e-05 | [1555486458178_18250](https://philly/#/job/eu2/ipgsrch/1555486458178_18250) | _best | 11.468 | [1555486458178_25732](https://philly/#/job/eu2/ipgsrch/1555486458178_25732) | --- | --- | --- |
| 0.1 | 0.3 | 0.2 | 0.0003 | 1e-05 | [1555486458178_18251](https://philly/#/job/eu2/ipgsrch/1555486458178_18251) | _best | 12.004 | [1555486458178_25733](https://philly/#/job/eu2/ipgsrch/1555486458178_25733) | --- | --- | --- |
| 0.1 | 0.3 | 0.0 | 0.0003 | 0.0001 | [1555486458178_18502](https://philly/#/job/eu2/ipgsrch/1555486458178_18502) | _best | 11.048 | [1555486458178_25734](https://philly/#/job/eu2/ipgsrch/1555486458178_25734) | --- | --- | --- |
| 0.1 | 0.3 | 0.1 | 0.0003 | 0.0001 | [1555486458178_18254](https://philly/#/job/eu2/ipgsrch/1555486458178_18254) | _best | 11.409 | [1555486458178_25735](https://philly/#/job/eu2/ipgsrch/1555486458178_25735) | --- | --- | --- |
| 0.1 | 0.3 | 0.2 | 0.0003 | 0.0001 | [1555486458178_18255](https://philly/#/job/eu2/ipgsrch/1555486458178_18255) | _best | 11.897 | [1555486458178_25736](https://philly/#/job/eu2/ipgsrch/1555486458178_25736) | --- | --- | --- |
| 0.2 | 0.3 | 0.0 | 0.0003 | 0.0 | [1555486458178_18505](https://philly/#/job/eu2/ipgsrch/1555486458178_18505) | _best | 10.959 | [1555486458178_25737](https://philly/#/job/eu2/ipgsrch/1555486458178_25737) | --- | --- | --- |
| 0.2 | 0.3 | 0.1 | 0.0003 | 0.0 | [1555486458178_18506](https://philly/#/job/eu2/ipgsrch/1555486458178_18506) | _best | 11.579 | [1555486458178_25738](https://philly/#/job/eu2/ipgsrch/1555486458178_25738) | --- | --- | --- |
| 0.2 | 0.3 | 0.2 | 0.0003 | 0.0 | [1555486458178_18507](https://philly/#/job/eu2/ipgsrch/1555486458178_18507) | _best | 12.036 | [1555486458178_25739](https://philly/#/job/eu2/ipgsrch/1555486458178_25739) | --- | --- | --- |
| 0.2 | 0.3 | 0.0 | 0.0003 | 1e-05 | [1555486458178_18508](https://philly/#/job/eu2/ipgsrch/1555486458178_18508) | _best | 11.239 | [1555486458178_25740](https://philly/#/job/eu2/ipgsrch/1555486458178_25740) | --- | --- | --- |
| 0.2 | 0.3 | 0.1 | 0.0003 | 1e-05 | [1555486458178_18260](https://philly/#/job/eu2/ipgsrch/1555486458178_18260) | _best | 12.281 | [1555486458178_25741](https://philly/#/job/eu2/ipgsrch/1555486458178_25741) | --- | --- | --- |
| 0.2 | 0.3 | 0.2 | 0.0003 | 1e-05 | [1555486458178_18261](https://philly/#/job/eu2/ipgsrch/1555486458178_18261) | _best | 12.036 | [1555486458178_25742](https://philly/#/job/eu2/ipgsrch/1555486458178_25742) | --- | --- | --- |
| 0.2 | 0.3 | 0.0 | 0.0003 | 0.0001 | [1555486458178_18511](https://philly/#/job/eu2/ipgsrch/1555486458178_18511) | _best | 11.188 | [1555486458178_25743](https://philly/#/job/eu2/ipgsrch/1555486458178_25743) | --- | --- | --- |
| 0.2 | 0.3 | 0.1 | 0.0003 | 0.0001 | [1555486458178_18263](https://philly/#/job/eu2/ipgsrch/1555486458178_18263) | _best | 11.399 | [1555486458178_25744](https://philly/#/job/eu2/ipgsrch/1555486458178_25744) | --- | --- | --- |
| 0.2 | 0.3 | 0.2 | 0.0003 | 0.0001 | [1555486458178_18264](https://philly/#/job/eu2/ipgsrch/1555486458178_18264) | _best | 12.118 | [1555486458178_25745](https://philly/#/job/eu2/ipgsrch/1555486458178_25745) | --- | --- | --- |
| 0.0 | 0.1 | 0.0 | 0.0004 | 0.0 | [1555486458178_18514](https://philly/#/job/eu2/ipgsrch/1555486458178_18514) | _best | 10.808 | [1555486458178_25746](https://philly/#/job/eu2/ipgsrch/1555486458178_25746) | --- | --- | --- |
| 0.0 | 0.1 | 0.1 | 0.0004 | 0.0 | [1555486458178_18515](https://philly/#/job/eu2/ipgsrch/1555486458178_18515) | _best | 11.416 | [1555486458178_25747](https://philly/#/job/eu2/ipgsrch/1555486458178_25747) | --- | --- | --- |
| 0.0 | 0.1 | 0.2 | 0.0004 | 0.0 | [1555486458178_18516](https://philly/#/job/eu2/ipgsrch/1555486458178_18516) | _best | 11.967 | [1555486458178_25748](https://philly/#/job/eu2/ipgsrch/1555486458178_25748) | --- | --- | --- |
| 0.0 | 0.1 | 0.0 | 0.0004 | 1e-05 | [1555486458178_18517](https://philly/#/job/eu2/ipgsrch/1555486458178_18517) | _best | 10.83 | [1555486458178_25749](https://philly/#/job/eu2/ipgsrch/1555486458178_25749) | --- | --- | --- |
| 0.0 | 0.1 | 0.1 | 0.0004 | 1e-05 | [1555486458178_18518](https://philly/#/job/eu2/ipgsrch/1555486458178_18518) | _best | 11.416 | [1555486458178_25750](https://philly/#/job/eu2/ipgsrch/1555486458178_25750) | --- | --- | --- |
| 0.0 | 0.1 | 0.2 | 0.0004 | 1e-05 | [1555486458178_18519](https://philly/#/job/eu2/ipgsrch/1555486458178_18519) | _best | 11.967 | [1555486458178_25751](https://philly/#/job/eu2/ipgsrch/1555486458178_25751) | --- | --- | --- |
| 0.0 | 0.1 | 0.0 | 0.0004 | 0.0001 | [1555486458178_18520](https://philly/#/job/eu2/ipgsrch/1555486458178_18520) | _best | 10.843 | [1555486458178_25752](https://philly/#/job/eu2/ipgsrch/1555486458178_25752) | --- | --- | --- |
| 0.0 | 0.1 | 0.1 | 0.0004 | 0.0001 | [1555486458178_18521](https://philly/#/job/eu2/ipgsrch/1555486458178_18521) | _best | 11.445 | [1555486458178_25753](https://philly/#/job/eu2/ipgsrch/1555486458178_25753) | --- | --- | --- |
| 0.0 | 0.1 | 0.2 | 0.0004 | 0.0001 | [1555486458178_18522](https://philly/#/job/eu2/ipgsrch/1555486458178_18522) | _best | 11.98 | [1555486458178_25754](https://philly/#/job/eu2/ipgsrch/1555486458178_25754) | --- | --- | --- |
| 0.1 | 0.1 | 0.0 | 0.0004 | 0.0 | [1555486458178_18523](https://philly/#/job/eu2/ipgsrch/1555486458178_18523) | _best | 10.787 | [1555486458178_25755](https://philly/#/job/eu2/ipgsrch/1555486458178_25755) | --- | --- | --- |
| 0.1 | 0.1 | 0.1 | 0.0004 | 0.0 | [1555486458178_18524](https://philly/#/job/eu2/ipgsrch/1555486458178_18524) | _best | 11.38 | [1555486458178_25756](https://philly/#/job/eu2/ipgsrch/1555486458178_25756) | --- | --- | --- |
| 0.1 | 0.1 | 0.2 | 0.0004 | 0.0 | [1555486458178_18525](https://philly/#/job/eu2/ipgsrch/1555486458178_18525) | _best | 12.001 | [1555486458178_25757](https://philly/#/job/eu2/ipgsrch/1555486458178_25757) | --- | --- | --- |
| 0.1 | 0.1 | 0.0 | 0.0004 | 1e-05 | [1555486458178_18526](https://philly/#/job/eu2/ipgsrch/1555486458178_18526) | _best | 10.787 | [1555486458178_25758](https://philly/#/job/eu2/ipgsrch/1555486458178_25758) | --- | --- | --- |
| 0.1 | 0.1 | 0.1 | 0.0004 | 1e-05 | [1555486458178_18278](https://philly/#/job/eu2/ipgsrch/1555486458178_18278) | _best | 11.373 | [1555486458178_25759](https://philly/#/job/eu2/ipgsrch/1555486458178_25759) | --- | --- | --- |
| 0.1 | 0.1 | 0.2 | 0.0004 | 1e-05 | [1555486458178_18279](https://philly/#/job/eu2/ipgsrch/1555486458178_18279) | _best | 11.93 | [1555486458178_25760](https://philly/#/job/eu2/ipgsrch/1555486458178_25760) | --- | --- | --- |
| 0.1 | 0.1 | 0.0 | 0.0004 | 0.0001 | [1555486458178_18529](https://philly/#/job/eu2/ipgsrch/1555486458178_18529) | _best | 10.795 | [1555486458178_25761](https://philly/#/job/eu2/ipgsrch/1555486458178_25761) | --- | --- | --- |
| 0.1 | 0.1 | 0.1 | 0.0004 | 0.0001 | [1555486458178_23270](https://philly/#/job/eu2/ipgsrch/1555486458178_23270) | _best | 11.403 | [1555486458178_25762](https://philly/#/job/eu2/ipgsrch/1555486458178_25762) | --- | --- | --- |
| 0.1 | 0.1 | 0.2 | 0.0004 | 0.0001 | [1555486458178_18282](https://philly/#/job/eu2/ipgsrch/1555486458178_18282) | _best | 11.934 | [1555486458178_25763](https://philly/#/job/eu2/ipgsrch/1555486458178_25763) | --- | --- | --- |
| 0.2 | 0.1 | 0.0 | 0.0004 | 0.0 | [1555486458178_18532](https://philly/#/job/eu2/ipgsrch/1555486458178_18532) | _best | 10.782 | [1555486458178_25764](https://philly/#/job/eu2/ipgsrch/1555486458178_25764) | --- | --- | --- |
| 0.2 | 0.1 | 0.1 | 0.0004 | 0.0 | [1555486458178_18533](https://philly/#/job/eu2/ipgsrch/1555486458178_18533) | _best | 11.381 | [1555486458178_25765](https://philly/#/job/eu2/ipgsrch/1555486458178_25765) | --- | --- | --- |
| 0.2 | 0.1 | 0.2 | 0.0004 | 0.0 | [1555486458178_23271](https://philly/#/job/eu2/ipgsrch/1555486458178_23271) | _best | 11.942 | [1555486458178_25766](https://philly/#/job/eu2/ipgsrch/1555486458178_25766) | --- | --- | --- |
| 0.2 | 0.1 | 0.0 | 0.0004 | 1e-05 | [1555486458178_18535](https://philly/#/job/eu2/ipgsrch/1555486458178_18535) | _best | 10.782 | [1555486458178_25767](https://philly/#/job/eu2/ipgsrch/1555486458178_25767) | --- | --- | --- |
| 0.2 | 0.1 | 0.1 | 0.0004 | 1e-05 | [1555486458178_18287](https://philly/#/job/eu2/ipgsrch/1555486458178_18287) | _best | 11.381 | [1555486458178_25768](https://philly/#/job/eu2/ipgsrch/1555486458178_25768) | --- | --- | --- |
| 0.2 | 0.1 | 0.2 | 0.0004 | 1e-05 | [1555486458178_18288](https://philly/#/job/eu2/ipgsrch/1555486458178_18288) | _best | 11.942 | [1555486458178_25769](https://philly/#/job/eu2/ipgsrch/1555486458178_25769) | --- | --- | --- |
| 0.2 | 0.1 | 0.0 | 0.0004 | 0.0001 | [1555486458178_18538](https://philly/#/job/eu2/ipgsrch/1555486458178_18538) | _best | 10.899 | [1555486458178_25770](https://philly/#/job/eu2/ipgsrch/1555486458178_25770) | --- | --- | --- |
| 0.2 | 0.1 | 0.1 | 0.0004 | 0.0001 | [1555486458178_18290](https://philly/#/job/eu2/ipgsrch/1555486458178_18290) | _best | 11.381 | [1555486458178_25771](https://philly/#/job/eu2/ipgsrch/1555486458178_25771) | --- | --- | --- |
| 0.2 | 0.1 | 0.2 | 0.0004 | 0.0001 | [1555486458178_18291](https://philly/#/job/eu2/ipgsrch/1555486458178_18291) | _best | 12.016 | [1555486458178_25772](https://philly/#/job/eu2/ipgsrch/1555486458178_25772) | --- | --- | --- |
| 0.0 | 0.2 | 0.0 | 0.0004 | 0.0 | [1555486458178_23272](https://philly/#/job/eu2/ipgsrch/1555486458178_23272) | _best | 10.839 | [1555486458178_25773](https://philly/#/job/eu2/ipgsrch/1555486458178_25773) | --- | --- | --- |
| 0.0 | 0.2 | 0.1 | 0.0004 | 0.0 | [1555486458178_18542](https://philly/#/job/eu2/ipgsrch/1555486458178_18542) | _best | 11.584 | [1555486458178_25774](https://philly/#/job/eu2/ipgsrch/1555486458178_25774) | --- | --- | --- |
| 0.0 | 0.2 | 0.2 | 0.0004 | 0.0 | [1555486458178_18543](https://philly/#/job/eu2/ipgsrch/1555486458178_18543) | _best | 11.969 | [1555486458178_25775](https://philly/#/job/eu2/ipgsrch/1555486458178_25775) | --- | --- | --- |
| 0.0 | 0.2 | 0.0 | 0.0004 | 1e-05 | [1555486458178_18544](https://philly/#/job/eu2/ipgsrch/1555486458178_18544) | _best | 10.839 | [1555486458178_25776](https://philly/#/job/eu2/ipgsrch/1555486458178_25776) | --- | --- | --- |
| 0.0 | 0.2 | 0.1 | 0.0004 | 1e-05 | [1555486458178_18545](https://philly/#/job/eu2/ipgsrch/1555486458178_18545) | _best | 11.545 | [1555486458178_25777](https://philly/#/job/eu2/ipgsrch/1555486458178_25777) | --- | --- | --- |
| 0.0 | 0.2 | 0.2 | 0.0004 | 1e-05 | [1555486458178_18546](https://philly/#/job/eu2/ipgsrch/1555486458178_18546) | _best | 11.948 | [1555486458178_25778](https://philly/#/job/eu2/ipgsrch/1555486458178_25778) | --- | --- | --- |
| 0.0 | 0.2 | 0.0 | 0.0004 | 0.0001 | [1555486458178_18547](https://philly/#/job/eu2/ipgsrch/1555486458178_18547) | _best | 10.83 | [1555486458178_25779](https://philly/#/job/eu2/ipgsrch/1555486458178_25779) | --- | --- | --- |
| 0.0 | 0.2 | 0.1 | 0.0004 | 0.0001 | [1555486458178_18548](https://philly/#/job/eu2/ipgsrch/1555486458178_18548) | _best | 11.853 | [1555486458178_25780](https://philly/#/job/eu2/ipgsrch/1555486458178_25780) | --- | --- | --- |
| 0.0 | 0.2 | 0.2 | 0.0004 | 0.0001 | [1555486458178_18549](https://philly/#/job/eu2/ipgsrch/1555486458178_18549) | _best | 11.969 | [1555486458178_25781](https://philly/#/job/eu2/ipgsrch/1555486458178_25781) | --- | --- | --- |
| 0.1 | 0.2 | 0.0 | 0.0004 | 0.0 | [1555486458178_18550](https://philly/#/job/eu2/ipgsrch/1555486458178_18550) | _best | 10.792 | [1555486458178_25782](https://philly/#/job/eu2/ipgsrch/1555486458178_25782) | --- | --- | --- |
| 0.1 | 0.2 | 0.1 | 0.0004 | 0.0 | [1555486458178_23273](https://philly/#/job/eu2/ipgsrch/1555486458178_23273) | _best | 11.392 | [1555486458178_25784](https://philly/#/job/eu2/ipgsrch/1555486458178_25784) | --- | --- | --- |
| 0.1 | 0.2 | 0.2 | 0.0004 | 0.0 | [1555486458178_18552](https://philly/#/job/eu2/ipgsrch/1555486458178_18552) | _best | 11.924 | [1555486458178_25785](https://philly/#/job/eu2/ipgsrch/1555486458178_25785) | --- | --- | --- |
| 0.1 | 0.2 | 0.0 | 0.0004 | 1e-05 | [1555486458178_18553](https://philly/#/job/eu2/ipgsrch/1555486458178_18553) | _best | 10.775 | [1555486458178_25786](https://philly/#/job/eu2/ipgsrch/1555486458178_25786) | --- | --- | --- |
| 0.1 | 0.2 | 0.1 | 0.0004 | 1e-05 | [1555486458178_18305](https://philly/#/job/eu2/ipgsrch/1555486458178_18305) | _best | 11.392 | [1555486458178_25787](https://philly/#/job/eu2/ipgsrch/1555486458178_25787) | --- | --- | --- |
| 0.1 | 0.2 | 0.2 | 0.0004 | 1e-05 | [1555486458178_18306](https://philly/#/job/eu2/ipgsrch/1555486458178_18306) | _best | 11.924 | [1555486458178_25788](https://philly/#/job/eu2/ipgsrch/1555486458178_25788) | --- | --- | --- |
| 0.1 | 0.2 | 0.0 | 0.0004 | 0.0001 | [1555486458178_18556](https://philly/#/job/eu2/ipgsrch/1555486458178_18556) | _best | 10.799 | [1555486458178_25789](https://philly/#/job/eu2/ipgsrch/1555486458178_25789) | --- | --- | --- |
| 0.1 | 0.2 | 0.1 | 0.0004 | 0.0001 | [1555486458178_18308](https://philly/#/job/eu2/ipgsrch/1555486458178_18308) | _best | 11.389 | [1555486458178_25790](https://philly/#/job/eu2/ipgsrch/1555486458178_25790) | --- | --- | --- |
| 0.1 | 0.2 | 0.2 | 0.0004 | 0.0001 | [1555486458178_18309](https://philly/#/job/eu2/ipgsrch/1555486458178_18309) | _best | 11.834 | [1555486458178_25791](https://philly/#/job/eu2/ipgsrch/1555486458178_25791) | --- | --- | --- |
| 0.2 | 0.2 | 0.0 | 0.0004 | 0.0 | [1555486458178_18559](https://philly/#/job/eu2/ipgsrch/1555486458178_18559) | _best | 10.789 | [1555486458178_25792](https://philly/#/job/eu2/ipgsrch/1555486458178_25792) | --- | --- | --- |
| 0.2 | 0.2 | 0.1 | 0.0004 | 0.0 | [1555486458178_18560](https://philly/#/job/eu2/ipgsrch/1555486458178_18560) | _best | 13.254 | [1555486458178_25793](https://philly/#/job/eu2/ipgsrch/1555486458178_25793) | --- | --- | --- |
| 0.2 | 0.2 | 0.2 | 0.0004 | 0.0 | [1555486458178_18562](https://philly/#/job/eu2/ipgsrch/1555486458178_18562) | _best | 11.94 | [1555486458178_25794](https://philly/#/job/eu2/ipgsrch/1555486458178_25794) | --- | --- | --- |
| 0.2 | 0.2 | 0.0 | 0.0004 | 1e-05 | [1555486458178_18563](https://philly/#/job/eu2/ipgsrch/1555486458178_18563) | _best | 10.794 | [1555486458178_25795](https://philly/#/job/eu2/ipgsrch/1555486458178_25795) | --- | --- | --- |
| 0.2 | 0.2 | 0.1 | 0.0004 | 1e-05 | [1555486458178_18314](https://philly/#/job/eu2/ipgsrch/1555486458178_18314) | _best | 11.389 | [1555486458178_25796](https://philly/#/job/eu2/ipgsrch/1555486458178_25796) | --- | --- | --- |
| 0.2 | 0.2 | 0.2 | 0.0004 | 1e-05 | [1555486458178_18315](https://philly/#/job/eu2/ipgsrch/1555486458178_18315) | _best | 11.948 | [1555486458178_25797](https://philly/#/job/eu2/ipgsrch/1555486458178_25797) | --- | --- | --- |
| 0.2 | 0.2 | 0.0 | 0.0004 | 0.0001 | [1555486458178_18566](https://philly/#/job/eu2/ipgsrch/1555486458178_18566) | _best | 10.782 | [1555486458178_25798](https://philly/#/job/eu2/ipgsrch/1555486458178_25798) | --- | --- | --- |
| 0.2 | 0.2 | 0.1 | 0.0004 | 0.0001 | [1555486458178_18317](https://philly/#/job/eu2/ipgsrch/1555486458178_18317) | _best | 11.372 | [1555486458178_25799](https://philly/#/job/eu2/ipgsrch/1555486458178_25799) | --- | --- | --- |
| 0.2 | 0.2 | 0.2 | 0.0004 | 0.0001 | [1555486458178_23274](https://philly/#/job/eu2/ipgsrch/1555486458178_23274) | _best | 11.929 | [1555486458178_25800](https://philly/#/job/eu2/ipgsrch/1555486458178_25800) | --- | --- | --- |
| 0.0 | 0.3 | 0.0 | 0.0004 | 0.0 | [1555486458178_18569](https://philly/#/job/eu2/ipgsrch/1555486458178_18569) | _best | 10.887 | [1555486458178_25801](https://philly/#/job/eu2/ipgsrch/1555486458178_25801) | --- | --- | --- |
| 0.0 | 0.3 | 0.1 | 0.0004 | 0.0 | [1555486458178_18570](https://philly/#/job/eu2/ipgsrch/1555486458178_18570) | _best | 11.43 | [1555486458178_25802](https://philly/#/job/eu2/ipgsrch/1555486458178_25802) | --- | --- | --- |
| 0.0 | 0.3 | 0.2 | 0.0004 | 0.0 | [1555486458178_18571](https://philly/#/job/eu2/ipgsrch/1555486458178_18571) | _best | 11.975 | [1555486458178_25803](https://philly/#/job/eu2/ipgsrch/1555486458178_25803) | --- | --- | --- |
| 0.0 | 0.3 | 0.0 | 0.0004 | 1e-05 | [1555486458178_18572](https://philly/#/job/eu2/ipgsrch/1555486458178_18572) | _best | 10.833 | [1555486458178_25804](https://philly/#/job/eu2/ipgsrch/1555486458178_25804) | --- | --- | --- |
| 0.0 | 0.3 | 0.1 | 0.0004 | 1e-05 | [1555486458178_18573](https://philly/#/job/eu2/ipgsrch/1555486458178_18573) | _best | 11.43 | [1555486458178_25805](https://philly/#/job/eu2/ipgsrch/1555486458178_25805) | --- | --- | --- |
| 0.0 | 0.3 | 0.2 | 0.0004 | 1e-05 | [1555486458178_18574](https://philly/#/job/eu2/ipgsrch/1555486458178_18574) | _best | 11.975 | [1555486458178_25806](https://philly/#/job/eu2/ipgsrch/1555486458178_25806) | --- | --- | --- |
| 0.0 | 0.3 | 0.0 | 0.0004 | 0.0001 | [1555486458178_18575](https://philly/#/job/eu2/ipgsrch/1555486458178_18575) | _best | 10.825 | [1555486458178_25807](https://philly/#/job/eu2/ipgsrch/1555486458178_25807) | --- | --- | --- |
| 0.0 | 0.3 | 0.1 | 0.0004 | 0.0001 | [1555486458178_18576](https://philly/#/job/eu2/ipgsrch/1555486458178_18576) | _best | 11.517 | [1555486458178_25808](https://philly/#/job/eu2/ipgsrch/1555486458178_25808) | --- | --- | --- |
| 0.0 | 0.3 | 0.2 | 0.0004 | 0.0001 | [1555486458178_18577](https://philly/#/job/eu2/ipgsrch/1555486458178_18577) | _best | 11.978 | [1555486458178_25809](https://philly/#/job/eu2/ipgsrch/1555486458178_25809) | --- | --- | --- |
| 0.1 | 0.3 | 0.0 | 0.0004 | 0.0 | [1555486458178_18578](https://philly/#/job/eu2/ipgsrch/1555486458178_18578) | _best | 10.784 | [1555486458178_25810](https://philly/#/job/eu2/ipgsrch/1555486458178_25810) | --- | --- | --- |
| 0.1 | 0.3 | 0.1 | 0.0004 | 0.0 | [1555486458178_18580](https://philly/#/job/eu2/ipgsrch/1555486458178_18580) | _best | 11.424 | [1555486458178_25811](https://philly/#/job/eu2/ipgsrch/1555486458178_25811) | --- | --- | --- |
| 0.1 | 0.3 | 0.2 | 0.0004 | 0.0 | [1555486458178_18581](https://philly/#/job/eu2/ipgsrch/1555486458178_18581) | _best | 11.923 | [1555486458178_25812](https://philly/#/job/eu2/ipgsrch/1555486458178_25812) | --- | --- | --- |
| 0.1 | 0.3 | 0.0 | 0.0004 | 1e-05 | [1555486458178_18582](https://philly/#/job/eu2/ipgsrch/1555486458178_18582) | _best | 10.784 | [1555486458178_25813](https://philly/#/job/eu2/ipgsrch/1555486458178_25813) | --- | --- | --- |
| 0.1 | 0.3 | 0.1 | 0.0004 | 1e-05 | [1555486458178_18332](https://philly/#/job/eu2/ipgsrch/1555486458178_18332) | _best | 11.383 | [1555486458178_25814](https://philly/#/job/eu2/ipgsrch/1555486458178_25814) | --- | --- | --- |
| 0.1 | 0.3 | 0.2 | 0.0004 | 1e-05 | [1555486458178_18333](https://philly/#/job/eu2/ipgsrch/1555486458178_18333) | _best | 11.923 | [1555486458178_25815](https://philly/#/job/eu2/ipgsrch/1555486458178_25815) | --- | --- | --- |
| 0.1 | 0.3 | 0.0 | 0.0004 | 0.0001 | [1555486458178_18585](https://philly/#/job/eu2/ipgsrch/1555486458178_18585) | _best | 10.781 | [1555486458178_25816](https://philly/#/job/eu2/ipgsrch/1555486458178_25816) | --- | --- | --- |
| 0.1 | 0.3 | 0.1 | 0.0004 | 0.0001 | [1555486458178_18335](https://philly/#/job/eu2/ipgsrch/1555486458178_18335) | _best | 11.382 | [1555486458178_25817](https://philly/#/job/eu2/ipgsrch/1555486458178_25817) | --- | --- | --- |
| 0.1 | 0.3 | 0.2 | 0.0004 | 0.0001 | [1555486458178_18336](https://philly/#/job/eu2/ipgsrch/1555486458178_18336) | _best | 11.92 | [1555486458178_25818](https://philly/#/job/eu2/ipgsrch/1555486458178_25818) | --- | --- | --- |
| 0.2 | 0.3 | 0.0 | 0.0004 | 0.0 | [1555486458178_18588](https://philly/#/job/eu2/ipgsrch/1555486458178_18588) | _best | 10.803 | [1555486458178_25819](https://philly/#/job/eu2/ipgsrch/1555486458178_25819) | --- | --- | --- |
| 0.2 | 0.3 | 0.1 | 0.0004 | 0.0 | [1555486458178_18589](https://philly/#/job/eu2/ipgsrch/1555486458178_18589) | _best | 11.397 | [1555486458178_25820](https://philly/#/job/eu2/ipgsrch/1555486458178_25820) | --- | --- | --- |
| 0.2 | 0.3 | 0.2 | 0.0004 | 0.0 | [1555486458178_18590](https://philly/#/job/eu2/ipgsrch/1555486458178_18590) | _best | 11.926 | [1555486458178_25821](https://philly/#/job/eu2/ipgsrch/1555486458178_25821) | --- | --- | --- |
| 0.2 | 0.3 | 0.0 | 0.0004 | 1e-05 | [1555486458178_18591](https://philly/#/job/eu2/ipgsrch/1555486458178_18591) | _best | 10.772 | [1555486458178_25822](https://philly/#/job/eu2/ipgsrch/1555486458178_25822) | --- | --- | --- |
| 0.2 | 0.3 | 0.1 | 0.0004 | 1e-05 | [1555486458178_18341](https://philly/#/job/eu2/ipgsrch/1555486458178_18341) | _best | 11.397 | [1555486458178_25823](https://philly/#/job/eu2/ipgsrch/1555486458178_25823) | --- | --- | --- |
| 0.2 | 0.3 | 0.2 | 0.0004 | 1e-05 | [1555486458178_18342](https://philly/#/job/eu2/ipgsrch/1555486458178_18342) | _best | 11.935 | [1555486458178_25824](https://philly/#/job/eu2/ipgsrch/1555486458178_25824) | --- | --- | --- |
| 0.2 | 0.3 | 0.0 | 0.0004 | 0.0001 | [1555486458178_18594](https://philly/#/job/eu2/ipgsrch/1555486458178_18594) | _best | 10.777 | [1555486458178_25825](https://philly/#/job/eu2/ipgsrch/1555486458178_25825) | --- | --- | --- |
| 0.2 | 0.3 | 0.1 | 0.0004 | 0.0001 | [1555486458178_18344](https://philly/#/job/eu2/ipgsrch/1555486458178_18344) | _best | 11.395 | [1555486458178_25826](https://philly/#/job/eu2/ipgsrch/1555486458178_25826) | --- | --- | --- |
| 0.2 | 0.3 | 0.2 | 0.0004 | 0.0001 | [1555486458178_18345](https://philly/#/job/eu2/ipgsrch/1555486458178_18345) | _best | 11.932 | [1555486458178_25827](https://philly/#/job/eu2/ipgsrch/1555486458178_25827) | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.1
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_20449](https://philly/#/job/eu2/ipgsrch/1555486458178_20449)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.1
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_20450](https://philly/#/job/eu2/ipgsrch/1555486458178_20450)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.1
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_20451](https://philly/#/job/eu2/ipgsrch/1555486458178_20451)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.1
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_20452](https://philly/#/job/eu2/ipgsrch/1555486458178_20452)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.1
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_20453](https://philly/#/job/eu2/ipgsrch/1555486458178_20453)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.1
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_20454](https://philly/#/job/eu2/ipgsrch/1555486458178_20454)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.1
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_20455](https://philly/#/job/eu2/ipgsrch/1555486458178_20455)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.1
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_20456](https://philly/#/job/eu2/ipgsrch/1555486458178_20456)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.1
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_20457](https://philly/#/job/eu2/ipgsrch/1555486458178_20457)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.1
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_20458](https://philly/#/job/eu2/ipgsrch/1555486458178_20458)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.1
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_20459](https://philly/#/job/eu2/ipgsrch/1555486458178_20459)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.1
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_20460](https://philly/#/job/eu2/ipgsrch/1555486458178_20460)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.1
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_20461](https://philly/#/job/eu2/ipgsrch/1555486458178_20461)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.1
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_20462](https://philly/#/job/eu2/ipgsrch/1555486458178_20462)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.1
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_20463](https://philly/#/job/eu2/ipgsrch/1555486458178_20463)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.1
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_20464](https://philly/#/job/eu2/ipgsrch/1555486458178_20464)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.1
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_20465](https://philly/#/job/eu2/ipgsrch/1555486458178_20465)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.1
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_20466](https://philly/#/job/eu2/ipgsrch/1555486458178_20466)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.1
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_20467](https://philly/#/job/eu2/ipgsrch/1555486458178_20467)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.1
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_20468](https://philly/#/job/eu2/ipgsrch/1555486458178_20468)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.1
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_20469](https://philly/#/job/eu2/ipgsrch/1555486458178_20469)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.1
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_20470](https://philly/#/job/eu2/ipgsrch/1555486458178_20470)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.1
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_20471](https://philly/#/job/eu2/ipgsrch/1555486458178_20471)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.1
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_20472](https://philly/#/job/eu2/ipgsrch/1555486458178_20472)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.1
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_20473](https://philly/#/job/eu2/ipgsrch/1555486458178_20473)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.1
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_20474](https://philly/#/job/eu2/ipgsrch/1555486458178_20474)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.1
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_20475](https://philly/#/job/eu2/ipgsrch/1555486458178_20475)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.2
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_20476](https://philly/#/job/eu2/ipgsrch/1555486458178_20476)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.2
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_20477](https://philly/#/job/eu2/ipgsrch/1555486458178_20477)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.2
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_20478](https://philly/#/job/eu2/ipgsrch/1555486458178_20478)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.2
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_20479](https://philly/#/job/eu2/ipgsrch/1555486458178_20479)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.2
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_20480](https://philly/#/job/eu2/ipgsrch/1555486458178_20480)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.2
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_20481](https://philly/#/job/eu2/ipgsrch/1555486458178_20481)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.2
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_20482](https://philly/#/job/eu2/ipgsrch/1555486458178_20482)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.2
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_20483](https://philly/#/job/eu2/ipgsrch/1555486458178_20483)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.2
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_20484](https://philly/#/job/eu2/ipgsrch/1555486458178_20484)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.2
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_20485](https://philly/#/job/eu2/ipgsrch/1555486458178_20485)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.2
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_20486](https://philly/#/job/eu2/ipgsrch/1555486458178_20486)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.2
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_20487](https://philly/#/job/eu2/ipgsrch/1555486458178_20487)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.2
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_20488](https://philly/#/job/eu2/ipgsrch/1555486458178_20488)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.2
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_20489](https://philly/#/job/eu2/ipgsrch/1555486458178_20489)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.2
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_20490](https://philly/#/job/eu2/ipgsrch/1555486458178_20490)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.2
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_20491](https://philly/#/job/eu2/ipgsrch/1555486458178_20491)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.2
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_20492](https://philly/#/job/eu2/ipgsrch/1555486458178_20492)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.2
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_20493](https://philly/#/job/eu2/ipgsrch/1555486458178_20493)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.2
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_20494](https://philly/#/job/eu2/ipgsrch/1555486458178_20494)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.2
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_20495](https://philly/#/job/eu2/ipgsrch/1555486458178_20495)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.2
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_20496](https://philly/#/job/eu2/ipgsrch/1555486458178_20496)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.2
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_20497](https://philly/#/job/eu2/ipgsrch/1555486458178_20497)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.2
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_20498](https://philly/#/job/eu2/ipgsrch/1555486458178_20498)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.2
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_20499](https://philly/#/job/eu2/ipgsrch/1555486458178_20499)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.2
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_20500](https://philly/#/job/eu2/ipgsrch/1555486458178_20500)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.2
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_20501](https://philly/#/job/eu2/ipgsrch/1555486458178_20501)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.2
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_20502](https://philly/#/job/eu2/ipgsrch/1555486458178_20502)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.3
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_20503](https://philly/#/job/eu2/ipgsrch/1555486458178_20503)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.3
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_20504](https://philly/#/job/eu2/ipgsrch/1555486458178_20504)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.3
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_20505](https://philly/#/job/eu2/ipgsrch/1555486458178_20505)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.3
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_20506](https://philly/#/job/eu2/ipgsrch/1555486458178_20506)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.3
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_20507](https://philly/#/job/eu2/ipgsrch/1555486458178_20507)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.3
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_20508](https://philly/#/job/eu2/ipgsrch/1555486458178_20508)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.3
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_20509](https://philly/#/job/eu2/ipgsrch/1555486458178_20509)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.3
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_20510](https://philly/#/job/eu2/ipgsrch/1555486458178_20510)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.3
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_20511](https://philly/#/job/eu2/ipgsrch/1555486458178_20511)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.3
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_20512](https://philly/#/job/eu2/ipgsrch/1555486458178_20512)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.3
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_20513](https://philly/#/job/eu2/ipgsrch/1555486458178_20513)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.3
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_20514](https://philly/#/job/eu2/ipgsrch/1555486458178_20514)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.3
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_20515](https://philly/#/job/eu2/ipgsrch/1555486458178_20515)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.3
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_20516](https://philly/#/job/eu2/ipgsrch/1555486458178_20516)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.3
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_20517](https://philly/#/job/eu2/ipgsrch/1555486458178_20517)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.3
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_20518](https://philly/#/job/eu2/ipgsrch/1555486458178_20518)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.3
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_20519](https://philly/#/job/eu2/ipgsrch/1555486458178_20519)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.3
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_20520](https://philly/#/job/eu2/ipgsrch/1555486458178_20520)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.3
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_20521](https://philly/#/job/eu2/ipgsrch/1555486458178_20521)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.3
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_20522](https://philly/#/job/eu2/ipgsrch/1555486458178_20522)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.3
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_20523](https://philly/#/job/eu2/ipgsrch/1555486458178_20523)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.3
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_20524](https://philly/#/job/eu2/ipgsrch/1555486458178_20524)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.3
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_20525](https://philly/#/job/eu2/ipgsrch/1555486458178_20525)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.3
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_20526](https://philly/#/job/eu2/ipgsrch/1555486458178_20526)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.3
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_20527](https://philly/#/job/eu2/ipgsrch/1555486458178_20527)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.3
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_20528](https://philly/#/job/eu2/ipgsrch/1555486458178_20528)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0001
--dropout 0.3
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_20529](https://philly/#/job/eu2/ipgsrch/1555486458178_20529)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.1
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_18349](https://philly/#/job/eu2/ipgsrch/1555486458178_18349)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.1
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_18350](https://philly/#/job/eu2/ipgsrch/1555486458178_18350)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.1
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_18351](https://philly/#/job/eu2/ipgsrch/1555486458178_18351)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.1
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_18352](https://philly/#/job/eu2/ipgsrch/1555486458178_18352)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.1
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_18353](https://philly/#/job/eu2/ipgsrch/1555486458178_18353)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.1
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_18354](https://philly/#/job/eu2/ipgsrch/1555486458178_18354)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.1
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_18355](https://philly/#/job/eu2/ipgsrch/1555486458178_18355)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.1
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_18356](https://philly/#/job/eu2/ipgsrch/1555486458178_18356)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.1
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_18357](https://philly/#/job/eu2/ipgsrch/1555486458178_18357)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.1
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_18358](https://philly/#/job/eu2/ipgsrch/1555486458178_18358)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.1
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_18359](https://philly/#/job/eu2/ipgsrch/1555486458178_18359)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.1
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_18360](https://philly/#/job/eu2/ipgsrch/1555486458178_18360)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.1
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_18361](https://philly/#/job/eu2/ipgsrch/1555486458178_18361)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.1
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_18115](https://philly/#/job/eu2/ipgsrch/1555486458178_18115)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.1
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_18116](https://philly/#/job/eu2/ipgsrch/1555486458178_18116)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.1
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_18364](https://philly/#/job/eu2/ipgsrch/1555486458178_18364)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.1
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_18118](https://philly/#/job/eu2/ipgsrch/1555486458178_18118)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.1
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_18119](https://philly/#/job/eu2/ipgsrch/1555486458178_18119)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.1
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_18367](https://philly/#/job/eu2/ipgsrch/1555486458178_18367)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.1
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_18368](https://philly/#/job/eu2/ipgsrch/1555486458178_18368)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.1
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_18369](https://philly/#/job/eu2/ipgsrch/1555486458178_18369)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.1
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_18370](https://philly/#/job/eu2/ipgsrch/1555486458178_18370)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.1
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_18124](https://philly/#/job/eu2/ipgsrch/1555486458178_18124)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.1
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_18125](https://philly/#/job/eu2/ipgsrch/1555486458178_18125)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.1
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_18373](https://philly/#/job/eu2/ipgsrch/1555486458178_18373)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.1
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_18127](https://philly/#/job/eu2/ipgsrch/1555486458178_18127)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.1
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_18128](https://philly/#/job/eu2/ipgsrch/1555486458178_18128)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.2
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_18376](https://philly/#/job/eu2/ipgsrch/1555486458178_18376)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.2
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_18377](https://philly/#/job/eu2/ipgsrch/1555486458178_18377)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.2
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_18378](https://philly/#/job/eu2/ipgsrch/1555486458178_18378)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.2
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_18379](https://philly/#/job/eu2/ipgsrch/1555486458178_18379)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.2
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_18380](https://philly/#/job/eu2/ipgsrch/1555486458178_18380)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.2
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_18381](https://philly/#/job/eu2/ipgsrch/1555486458178_18381)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.2
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_18382](https://philly/#/job/eu2/ipgsrch/1555486458178_18382)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.2
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_18383](https://philly/#/job/eu2/ipgsrch/1555486458178_18383)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.2
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_18384](https://philly/#/job/eu2/ipgsrch/1555486458178_18384)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.2
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_18385](https://philly/#/job/eu2/ipgsrch/1555486458178_18385)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.2
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_18386](https://philly/#/job/eu2/ipgsrch/1555486458178_18386)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.2
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_18387](https://philly/#/job/eu2/ipgsrch/1555486458178_18387)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.2
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_18388](https://philly/#/job/eu2/ipgsrch/1555486458178_18388)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.2
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_18142](https://philly/#/job/eu2/ipgsrch/1555486458178_18142)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.2
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_18143](https://philly/#/job/eu2/ipgsrch/1555486458178_18143)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.2
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_18391](https://philly/#/job/eu2/ipgsrch/1555486458178_18391)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.2
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_18145](https://philly/#/job/eu2/ipgsrch/1555486458178_18145)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.2
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_18146](https://philly/#/job/eu2/ipgsrch/1555486458178_18146)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.2
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_18394](https://philly/#/job/eu2/ipgsrch/1555486458178_18394)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.2
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_18395](https://philly/#/job/eu2/ipgsrch/1555486458178_18395)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.2
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_18396](https://philly/#/job/eu2/ipgsrch/1555486458178_18396)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.2
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_18397](https://philly/#/job/eu2/ipgsrch/1555486458178_18397)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.2
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_18151](https://philly/#/job/eu2/ipgsrch/1555486458178_18151)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.2
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_18152](https://philly/#/job/eu2/ipgsrch/1555486458178_18152)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.2
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_18400](https://philly/#/job/eu2/ipgsrch/1555486458178_18400)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.2
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_18154](https://philly/#/job/eu2/ipgsrch/1555486458178_18154)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.2
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_18155](https://philly/#/job/eu2/ipgsrch/1555486458178_18155)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.3
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_18403](https://philly/#/job/eu2/ipgsrch/1555486458178_18403)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.3
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_18404](https://philly/#/job/eu2/ipgsrch/1555486458178_18404)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.3
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_18405](https://philly/#/job/eu2/ipgsrch/1555486458178_18405)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.3
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_18406](https://philly/#/job/eu2/ipgsrch/1555486458178_18406)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.3
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_18407](https://philly/#/job/eu2/ipgsrch/1555486458178_18407)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.3
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_18408](https://philly/#/job/eu2/ipgsrch/1555486458178_18408)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.3
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_18409](https://philly/#/job/eu2/ipgsrch/1555486458178_18409)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.3
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_18410](https://philly/#/job/eu2/ipgsrch/1555486458178_18410)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.3
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_18411](https://philly/#/job/eu2/ipgsrch/1555486458178_18411)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.3
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_18412](https://philly/#/job/eu2/ipgsrch/1555486458178_18412)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.3
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_18413](https://philly/#/job/eu2/ipgsrch/1555486458178_18413)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.3
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_18414](https://philly/#/job/eu2/ipgsrch/1555486458178_18414)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.3
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_18415](https://philly/#/job/eu2/ipgsrch/1555486458178_18415)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.3
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_18169](https://philly/#/job/eu2/ipgsrch/1555486458178_18169)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.3
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_18170](https://philly/#/job/eu2/ipgsrch/1555486458178_18170)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.3
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_18418](https://philly/#/job/eu2/ipgsrch/1555486458178_18418)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.3
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_18172](https://philly/#/job/eu2/ipgsrch/1555486458178_18172)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.3
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_18173](https://philly/#/job/eu2/ipgsrch/1555486458178_18173)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.3
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_18421](https://philly/#/job/eu2/ipgsrch/1555486458178_18421)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.3
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_18422](https://philly/#/job/eu2/ipgsrch/1555486458178_18422)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.3
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_18423](https://philly/#/job/eu2/ipgsrch/1555486458178_18423)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.3
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_18424](https://philly/#/job/eu2/ipgsrch/1555486458178_18424)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.3
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_18178](https://philly/#/job/eu2/ipgsrch/1555486458178_18178)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.3
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_18179](https://philly/#/job/eu2/ipgsrch/1555486458178_18179)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.3
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_18427](https://philly/#/job/eu2/ipgsrch/1555486458178_18427)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.3
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_18181](https://philly/#/job/eu2/ipgsrch/1555486458178_18181)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0002
--dropout 0.3
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_18182](https://philly/#/job/eu2/ipgsrch/1555486458178_18182)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.1
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_18430](https://philly/#/job/eu2/ipgsrch/1555486458178_18430)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.1
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_18431](https://philly/#/job/eu2/ipgsrch/1555486458178_18431)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.1
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_18432](https://philly/#/job/eu2/ipgsrch/1555486458178_18432)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.1
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_18433](https://philly/#/job/eu2/ipgsrch/1555486458178_18433)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.1
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_18434](https://philly/#/job/eu2/ipgsrch/1555486458178_18434)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.1
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_18435](https://philly/#/job/eu2/ipgsrch/1555486458178_18435)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.1
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_18436](https://philly/#/job/eu2/ipgsrch/1555486458178_18436)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.1
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_18437](https://philly/#/job/eu2/ipgsrch/1555486458178_18437)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.1
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_18438](https://philly/#/job/eu2/ipgsrch/1555486458178_18438)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.1
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_18439](https://philly/#/job/eu2/ipgsrch/1555486458178_18439)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.1
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_18440](https://philly/#/job/eu2/ipgsrch/1555486458178_18440)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.1
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_18442](https://philly/#/job/eu2/ipgsrch/1555486458178_18442)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.1
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_18443](https://philly/#/job/eu2/ipgsrch/1555486458178_18443)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.1
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_18196](https://philly/#/job/eu2/ipgsrch/1555486458178_18196)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.1
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_18197](https://philly/#/job/eu2/ipgsrch/1555486458178_18197)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.1
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_18446](https://philly/#/job/eu2/ipgsrch/1555486458178_18446)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.1
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_18199](https://philly/#/job/eu2/ipgsrch/1555486458178_18199)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.1
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_18200](https://philly/#/job/eu2/ipgsrch/1555486458178_18200)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.1
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_18449](https://philly/#/job/eu2/ipgsrch/1555486458178_18449)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.1
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_18450](https://philly/#/job/eu2/ipgsrch/1555486458178_18450)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.1
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_18451](https://philly/#/job/eu2/ipgsrch/1555486458178_18451)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.1
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_18452](https://philly/#/job/eu2/ipgsrch/1555486458178_18452)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.1
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_18205](https://philly/#/job/eu2/ipgsrch/1555486458178_18205)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.1
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_18206](https://philly/#/job/eu2/ipgsrch/1555486458178_18206)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.1
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_18455](https://philly/#/job/eu2/ipgsrch/1555486458178_18455)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.1
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_18208](https://philly/#/job/eu2/ipgsrch/1555486458178_18208)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.1
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_18209](https://philly/#/job/eu2/ipgsrch/1555486458178_18209)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.2
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_18459](https://philly/#/job/eu2/ipgsrch/1555486458178_18459)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.2
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_18460](https://philly/#/job/eu2/ipgsrch/1555486458178_18460)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.2
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_18461](https://philly/#/job/eu2/ipgsrch/1555486458178_18461)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.2
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_18462](https://philly/#/job/eu2/ipgsrch/1555486458178_18462)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.2
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_18463](https://philly/#/job/eu2/ipgsrch/1555486458178_18463)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.2
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_18464](https://philly/#/job/eu2/ipgsrch/1555486458178_18464)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.2
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_18465](https://philly/#/job/eu2/ipgsrch/1555486458178_18465)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.2
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_18466](https://philly/#/job/eu2/ipgsrch/1555486458178_18466)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.2
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_18467](https://philly/#/job/eu2/ipgsrch/1555486458178_18467)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.2
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_18468](https://philly/#/job/eu2/ipgsrch/1555486458178_18468)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.2
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_18469](https://philly/#/job/eu2/ipgsrch/1555486458178_18469)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.2
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_18470](https://philly/#/job/eu2/ipgsrch/1555486458178_18470)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.2
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_18471](https://philly/#/job/eu2/ipgsrch/1555486458178_18471)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.2
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_18223](https://philly/#/job/eu2/ipgsrch/1555486458178_18223)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.2
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_18224](https://philly/#/job/eu2/ipgsrch/1555486458178_18224)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.2
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_18474](https://philly/#/job/eu2/ipgsrch/1555486458178_18474)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.2
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_18226](https://philly/#/job/eu2/ipgsrch/1555486458178_18226)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.2
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_18227](https://philly/#/job/eu2/ipgsrch/1555486458178_18227)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.2
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_18477](https://philly/#/job/eu2/ipgsrch/1555486458178_18477)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.2
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_18478](https://philly/#/job/eu2/ipgsrch/1555486458178_18478)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.2
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_18479](https://philly/#/job/eu2/ipgsrch/1555486458178_18479)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.2
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_18480](https://philly/#/job/eu2/ipgsrch/1555486458178_18480)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.2
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_18232](https://philly/#/job/eu2/ipgsrch/1555486458178_18232)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.2
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_18233](https://philly/#/job/eu2/ipgsrch/1555486458178_18233)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.2
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_18483](https://philly/#/job/eu2/ipgsrch/1555486458178_18483)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.2
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_18235](https://philly/#/job/eu2/ipgsrch/1555486458178_18235)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.2
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_18236](https://philly/#/job/eu2/ipgsrch/1555486458178_18236)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.3
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_18486](https://philly/#/job/eu2/ipgsrch/1555486458178_18486)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.3
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_18488](https://philly/#/job/eu2/ipgsrch/1555486458178_18488)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.3
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_18489](https://philly/#/job/eu2/ipgsrch/1555486458178_18489)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.3
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_18490](https://philly/#/job/eu2/ipgsrch/1555486458178_18490)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.3
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_18491](https://philly/#/job/eu2/ipgsrch/1555486458178_18491)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.3
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_18492](https://philly/#/job/eu2/ipgsrch/1555486458178_18492)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.3
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_18493](https://philly/#/job/eu2/ipgsrch/1555486458178_18493)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.3
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_18494](https://philly/#/job/eu2/ipgsrch/1555486458178_18494)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.3
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_18495](https://philly/#/job/eu2/ipgsrch/1555486458178_18495)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.3
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_18496](https://philly/#/job/eu2/ipgsrch/1555486458178_18496)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.3
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_18497](https://philly/#/job/eu2/ipgsrch/1555486458178_18497)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.3
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_18498](https://philly/#/job/eu2/ipgsrch/1555486458178_18498)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.3
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_18499](https://philly/#/job/eu2/ipgsrch/1555486458178_18499)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.3
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_18250](https://philly/#/job/eu2/ipgsrch/1555486458178_18250)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.3
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_18251](https://philly/#/job/eu2/ipgsrch/1555486458178_18251)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.3
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_18502](https://philly/#/job/eu2/ipgsrch/1555486458178_18502)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.3
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_18254](https://philly/#/job/eu2/ipgsrch/1555486458178_18254)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.3
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_18255](https://philly/#/job/eu2/ipgsrch/1555486458178_18255)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.3
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_18505](https://philly/#/job/eu2/ipgsrch/1555486458178_18505)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.3
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_18506](https://philly/#/job/eu2/ipgsrch/1555486458178_18506)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.3
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_18507](https://philly/#/job/eu2/ipgsrch/1555486458178_18507)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.3
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_18508](https://philly/#/job/eu2/ipgsrch/1555486458178_18508)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.3
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_18260](https://philly/#/job/eu2/ipgsrch/1555486458178_18260)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.3
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_18261](https://philly/#/job/eu2/ipgsrch/1555486458178_18261)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.3
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_18511](https://philly/#/job/eu2/ipgsrch/1555486458178_18511)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.3
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_18263](https://philly/#/job/eu2/ipgsrch/1555486458178_18263)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0003
--dropout 0.3
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_18264](https://philly/#/job/eu2/ipgsrch/1555486458178_18264)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.1
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_18514](https://philly/#/job/eu2/ipgsrch/1555486458178_18514)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.1
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_18515](https://philly/#/job/eu2/ipgsrch/1555486458178_18515)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.1
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_18516](https://philly/#/job/eu2/ipgsrch/1555486458178_18516)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.1
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_18517](https://philly/#/job/eu2/ipgsrch/1555486458178_18517)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.1
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_18518](https://philly/#/job/eu2/ipgsrch/1555486458178_18518)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.1
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_18519](https://philly/#/job/eu2/ipgsrch/1555486458178_18519)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.1
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_18520](https://philly/#/job/eu2/ipgsrch/1555486458178_18520)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.1
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_18521](https://philly/#/job/eu2/ipgsrch/1555486458178_18521)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.1
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_18522](https://philly/#/job/eu2/ipgsrch/1555486458178_18522)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.1
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_18523](https://philly/#/job/eu2/ipgsrch/1555486458178_18523)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.1
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_18524](https://philly/#/job/eu2/ipgsrch/1555486458178_18524)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.1
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_18525](https://philly/#/job/eu2/ipgsrch/1555486458178_18525)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.1
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_18526](https://philly/#/job/eu2/ipgsrch/1555486458178_18526)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.1
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_18278](https://philly/#/job/eu2/ipgsrch/1555486458178_18278)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.1
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_18279](https://philly/#/job/eu2/ipgsrch/1555486458178_18279)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.1
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_18529](https://philly/#/job/eu2/ipgsrch/1555486458178_18529)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.1
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_18281](https://philly/#/job/eu2/ipgsrch/1555486458178_18281)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.1
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_18282](https://philly/#/job/eu2/ipgsrch/1555486458178_18282)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.1
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_18532](https://philly/#/job/eu2/ipgsrch/1555486458178_18532)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.1
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_18533](https://philly/#/job/eu2/ipgsrch/1555486458178_18533)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.1
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_18534](https://philly/#/job/eu2/ipgsrch/1555486458178_18534)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.1
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_18535](https://philly/#/job/eu2/ipgsrch/1555486458178_18535)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.1
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_18287](https://philly/#/job/eu2/ipgsrch/1555486458178_18287)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.1
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_18288](https://philly/#/job/eu2/ipgsrch/1555486458178_18288)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.1
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_18538](https://philly/#/job/eu2/ipgsrch/1555486458178_18538)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.1
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_18290](https://philly/#/job/eu2/ipgsrch/1555486458178_18290)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.1
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_18291](https://philly/#/job/eu2/ipgsrch/1555486458178_18291)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.2
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_18541](https://philly/#/job/eu2/ipgsrch/1555486458178_18541)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.2
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_18542](https://philly/#/job/eu2/ipgsrch/1555486458178_18542)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.2
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_18543](https://philly/#/job/eu2/ipgsrch/1555486458178_18543)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.2
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_18544](https://philly/#/job/eu2/ipgsrch/1555486458178_18544)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.2
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_18545](https://philly/#/job/eu2/ipgsrch/1555486458178_18545)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.2
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_18546](https://philly/#/job/eu2/ipgsrch/1555486458178_18546)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.2
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_18547](https://philly/#/job/eu2/ipgsrch/1555486458178_18547)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.2
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_18548](https://philly/#/job/eu2/ipgsrch/1555486458178_18548)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.2
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_18549](https://philly/#/job/eu2/ipgsrch/1555486458178_18549)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.2
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_18550](https://philly/#/job/eu2/ipgsrch/1555486458178_18550)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.2
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_18551](https://philly/#/job/eu2/ipgsrch/1555486458178_18551)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.2
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_18552](https://philly/#/job/eu2/ipgsrch/1555486458178_18552)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.2
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_18553](https://philly/#/job/eu2/ipgsrch/1555486458178_18553)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.2
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_18305](https://philly/#/job/eu2/ipgsrch/1555486458178_18305)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.2
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_18306](https://philly/#/job/eu2/ipgsrch/1555486458178_18306)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.2
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_18556](https://philly/#/job/eu2/ipgsrch/1555486458178_18556)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.2
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_18308](https://philly/#/job/eu2/ipgsrch/1555486458178_18308)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.2
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_18309](https://philly/#/job/eu2/ipgsrch/1555486458178_18309)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.2
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_18559](https://philly/#/job/eu2/ipgsrch/1555486458178_18559)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.2
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_18560](https://philly/#/job/eu2/ipgsrch/1555486458178_18560)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.2
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_18562](https://philly/#/job/eu2/ipgsrch/1555486458178_18562)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.2
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_18563](https://philly/#/job/eu2/ipgsrch/1555486458178_18563)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.2
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_18314](https://philly/#/job/eu2/ipgsrch/1555486458178_18314)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.2
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_18315](https://philly/#/job/eu2/ipgsrch/1555486458178_18315)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.2
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_18566](https://philly/#/job/eu2/ipgsrch/1555486458178_18566)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.2
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_18317](https://philly/#/job/eu2/ipgsrch/1555486458178_18317)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.2
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_18318](https://philly/#/job/eu2/ipgsrch/1555486458178_18318)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.3
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_18569](https://philly/#/job/eu2/ipgsrch/1555486458178_18569)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.3
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_18570](https://philly/#/job/eu2/ipgsrch/1555486458178_18570)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.3
--clip-norm 0.0
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_18571](https://philly/#/job/eu2/ipgsrch/1555486458178_18571)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.3
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_18572](https://philly/#/job/eu2/ipgsrch/1555486458178_18572)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.3
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_18573](https://philly/#/job/eu2/ipgsrch/1555486458178_18573)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.3
--clip-norm 0.0
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_18574](https://philly/#/job/eu2/ipgsrch/1555486458178_18574)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.3
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_18575](https://philly/#/job/eu2/ipgsrch/1555486458178_18575)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.3
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_18576](https://philly/#/job/eu2/ipgsrch/1555486458178_18576)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.3
--clip-norm 0.0
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_18577](https://philly/#/job/eu2/ipgsrch/1555486458178_18577)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.3
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_18578](https://philly/#/job/eu2/ipgsrch/1555486458178_18578)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.3
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_18580](https://philly/#/job/eu2/ipgsrch/1555486458178_18580)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.3
--clip-norm 0.1
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_18581](https://philly/#/job/eu2/ipgsrch/1555486458178_18581)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.3
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_18582](https://philly/#/job/eu2/ipgsrch/1555486458178_18582)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.3
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_18332](https://philly/#/job/eu2/ipgsrch/1555486458178_18332)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.3
--clip-norm 0.1
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_18333](https://philly/#/job/eu2/ipgsrch/1555486458178_18333)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.3
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_18585](https://philly/#/job/eu2/ipgsrch/1555486458178_18585)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.3
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_18335](https://philly/#/job/eu2/ipgsrch/1555486458178_18335)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.3
--clip-norm 0.1
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_18336](https://philly/#/job/eu2/ipgsrch/1555486458178_18336)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.3
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.0
```
> job: [1555486458178_18588](https://philly/#/job/eu2/ipgsrch/1555486458178_18588)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.3
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.1
```
> job: [1555486458178_18589](https://philly/#/job/eu2/ipgsrch/1555486458178_18589)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.3
--clip-norm 0.2
--weight-decay 0.0
--label-smoothing 0.2
```
> job: [1555486458178_18590](https://philly/#/job/eu2/ipgsrch/1555486458178_18590)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.3
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.0
```
> job: [1555486458178_18591](https://philly/#/job/eu2/ipgsrch/1555486458178_18591)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.3
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.1
```
> job: [1555486458178_18341](https://philly/#/job/eu2/ipgsrch/1555486458178_18341)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.3
--clip-norm 0.2
--weight-decay 1e-05
--label-smoothing 0.2
```
> job: [1555486458178_18342](https://philly/#/job/eu2/ipgsrch/1555486458178_18342)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.3
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.0
```
> job: [1555486458178_18594](https://philly/#/job/eu2/ipgsrch/1555486458178_18594)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.3
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.1
```
> job: [1555486458178_18344](https://philly/#/job/eu2/ipgsrch/1555486458178_18344)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

> hyper parameter
```
--lr 0.0004
--dropout 0.3
--clip-norm 0.2
--weight-decay 0.0001
--label-smoothing 0.2
```
> job: [1555486458178_18345](https://philly/#/job/eu2/ipgsrch/1555486458178_18345)

| test | epoch | rouge-1 | rouge-2 | rouge-l |
| --- | --- | --- | --- | --- |

