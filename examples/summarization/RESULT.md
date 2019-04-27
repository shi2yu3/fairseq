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
python $rootdir/train.py $rootdir/$datadir -s src -t tgt --max-tokens 4000 -a $arch --share-all-embeddings --dropout 0.3 --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --weight-decay 0.0001 --lr 0.0005 --warmup-init-lr 1e-07 --warmup-updates 4000 --lr-scheduler inverse_sqrt --min-lr 1e-09 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-update 300000 --save-dir $PHILLY_JOB_DIRECTORY

# test commandLine
python $rootdir/generate.py $rootdir/$datadir --path $modelpath$model/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe --no-repeat-ngram-size 3
```
| train | test | model | result |
| --- | --- | --- | --- |
| [1555486458178_6374](https://philly/#/job/eu2/ipgsrch/1555486458178_6374) | [xxx](https://philly/#/job/eu2/ipgsrch/xxx) | transformer_vaswani_wmt_en_de_big | xxx |


# Parameter tuning


## Baseline

* train: [1555486458178_12550](https://philly/#/job/eu2/ipgsrch/1555486458178_12550)
```
lr = 0.0005
dropout = 0.3
weight-decay = 0.0001
label-smoothing = 0.1
```
| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |


## Learning rate tuning

* train: [1555486458178_12556](https://philly/#/job/eu2/ipgsrch/1555486458178_12556)
```
lr = 0.005
```
| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |

* train: [1555486458178_13511](https://philly/#/job/eu2/ipgsrch/1555486458178_13511)
```
lr = 0.00009
```
| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |

* train: [1555486458178_13510](https://philly/#/job/eu2/ipgsrch/1555486458178_13510)
```
lr = 0.00007
```
| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |

* train: [1555486458178_12557](https://philly/#/job/eu2/ipgsrch/1555486458178_12557)
```
lr = 0.00005
```
| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |

* train: [1555486458178_13508](https://philly/#/job/eu2/ipgsrch/1555486458178_13508)
```
lr = 0.00003
```
| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |

* train: [1555486458178_13509](https://philly/#/job/eu2/ipgsrch/1555486458178_13509)
```
lr = 0.00001
```
| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |


## Dropout rate tuning

* train: [1555486458178_12831](https://philly/#/job/eu2/ipgsrch/1555486458178_12831)
```
dropout = 0.0
```
| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |

* train: [1555486458178_12857](https://philly/#/job/eu2/ipgsrch/1555486458178_12857)
```
dropout = 0.1
```
| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |

* train: [1555486458178_12861](https://philly/#/job/eu2/ipgsrch/1555486458178_12861)
```
dropout = 0.2
```
| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |


## Weight decay tuning

* train: [1555486458178_12865](https://philly/#/job/eu2/ipgsrch/1555486458178_12865)
```
weight-decay = 0.00001
```
| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |

* train: [1555486458178_12866](https://philly/#/job/eu2/ipgsrch/1555486458178_12866)
```
weight-decay = 0.00005
```
| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |


## Label smoothing tuning

* train: [1555486458178_12880](https://philly/#/job/eu2/ipgsrch/1555486458178_12880)
```
label-smoothing = 0.01
```
| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |

* train: [1555486458178_12881](https://philly/#/job/eu2/ipgsrch/1555486458178_12881)
```
label-smoothing = 0.05
```
| test | epoch | rouge-1 | rouge-2 | rouge-l | 
| --- | --- | --- | --- | --- |



