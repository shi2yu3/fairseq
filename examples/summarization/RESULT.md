# Philly config

```
# "environmentVariables": {
    "rootdir": "/philly/eu2/ipgsrch/yushi/fairseq",
    "datadir": "data-bin/cnndm",
    "arch": "transformer_vaswani_wmt_en_de_big",
    "modelpath": "/var/storage/shared/ipgsrch/sys/jobs/application_",
    "model": "1555486458178_6374"
  },

# train commandLine
python $rootdir/train.py $rootdir/$datadir -s src -t tgt --max-tokens 4000 -a $arch --share-all-embeddings --dropout 0.3 --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --weight-decay 0.0001 --lr 0.0005 --warmup-init-lr 1e-07 --warmup-updates 4000 --lr-scheduler inverse_sqrt --min-lr 1e-09 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-update 30000000 --save-dir $PHILLY_JOB_DIRECTORY

# test commandLine
python $rootdir/generate.py $rootdir/$datadir --path $modelpath$model/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe --no-repeat-ngram-size 3
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
```
> train: [1555486458178_12550](https://philly/#/job/eu2/ipgsrch/1555486458178_12550)

| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |


## Learning rate tuning

* **lr = 0.05**
> train: [1555486458178_12556](https://philly/#/job/eu2/ipgsrch/1555486458178_12556)

| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |

* **lr = 0.00009**
> train: [1555486458178_13519](https://philly/#/job/eu2/ipgsrch/1555486458178_13519)

| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |

* **lr = 0.00007**
> train: [1555486458178_13518](https://philly/#/job/eu2/ipgsrch/1555486458178_13518)

| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |

* **lr = 0.00005**
> train: [1555486458178_13517](https://philly/#/job/eu2/ipgsrch/1555486458178_13517)
| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |

* **lr = 0.00003**
> train: [1555486458178_13516](https://philly/#/job/eu2/ipgsrch/1555486458178_13516)

| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |

* **lr = 0.00001**
> train: [1555486458178_13515](https://philly/#/job/eu2/ipgsrch/1555486458178_13515)

| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |


## Dropout rate tuning

* **dropout = 0.0**
> train: [1555486458178_12831](https://philly/#/job/eu2/ipgsrch/1555486458178_12831)

| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |

* **dropout = 0.1**
> train: [1555486458178_12857](https://philly/#/job/eu2/ipgsrch/1555486458178_12857)

| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |

* **dropout = 0.2**
> train: [1555486458178_12861](https://philly/#/job/eu2/ipgsrch/1555486458178_12861)

| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |


## Weight decay tuning

* **weight-decay = 0.00001**
> train: [1555486458178_12865](https://philly/#/job/eu2/ipgsrch/1555486458178_12865)

| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |

* **weight-decay = 0.00005**
> train: [1555486458178_12866](https://philly/#/job/eu2/ipgsrch/1555486458178_12866)

| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |


## Label smoothing tuning

* **label-smoothing = 0.01**
> train: [1555486458178_12880](https://philly/#/job/eu2/ipgsrch/1555486458178_12880)

| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |

* **label-smoothing = 0.05**
> train: [1555486458178_12881](https://philly/#/job/eu2/ipgsrch/1555486458178_12881)

| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |



