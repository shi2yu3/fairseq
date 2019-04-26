# Data processing

This need be run in folder examples/summarization

## Generate raw data by following https://github.com/shi2yu3/BertSum/PROCESS.md, the copy data to this folder
```
mkdir fairseq_data/
cp -r BertSum/fairseq_data/trunc400/ fairseq_data/
```

## Download tools

```
git clone https://github.com/rsennrich/subword-nmt.git
```

## Generate BPE code
```
cat fairseq_data/trunc400/train.src.txt fairseq_data/trunc400/train.tgt.txt > fairseq_data/trunc400/train.txt
python subword-nmt/learn_bpe.py -s 30000 < fairseq_data/trunc400/train.txt > fairseq_data/trunc400/code
```

## Tokenization
```
python subword-nmt/apply_bpe.py -c fairseq_data/trunc400/code < fairseq_data/trunc400/train.src.txt > fairseq_data/trunc400/train.src
python subword-nmt/apply_bpe.py -c fairseq_data/trunc400/code < fairseq_data/trunc400/train.tgt.txt > fairseq_data/trunc400/train.tgt
python subword-nmt/apply_bpe.py -c fairseq_data/trunc400/code < fairseq_data/trunc400/test.src.txt > fairseq_data/trunc400/test.src
python subword-nmt/apply_bpe.py -c fairseq_data/trunc400/code < fairseq_data/trunc400/test.tgt.txt > fairseq_data/trunc400/test.tgt
python subword-nmt/apply_bpe.py -c fairseq_data/trunc400/code < fairseq_data/trunc400/valid.src.txt > fairseq_data/trunc400/valid.src
python subword-nmt/apply_bpe.py -c fairseq_data/trunc400/code < fairseq_data/trunc400/valid.tgt.txt > fairseq_data/trunc400/valid.tgt
```

# Training

This need be run in the root folder

## Binarize the dataset

```
docker run --rm -it -v $(pwd):/workspace bertsum
python preprocess.py --source-lang src --target-lang tgt --joined-dictionary --trainpref examples/summarization/fairseq_data/trunc400/train --validpref examples/summarization/fairseq_data/trunc400/valid --testpref examples/summarization/fairseq_data/trunc400/test --destdir data-bin/cnndm
```
Output
```
| [src] Dictionary: 30391 types
| [src] examples/summarization/fairseq_data/trunc400/train.src: 287227 sents, 118030308 tokens, 0.0% replaced by <unk>
| [src] Dictionary: 30391 types
| [src] examples/summarization/fairseq_data/trunc400/valid.src: 13368 sents, 5451317 tokens, 0.000147% replaced by <unk>
| [src] Dictionary: 30391 types
| [src] examples/summarization/fairseq_data/trunc400/test.src: 11490 sents, 4700782 tokens, 0.000425% replaced by <unk>
| [tgt] Dictionary: 30391 types
| [tgt] examples/summarization/fairseq_data/trunc400/train.tgt: 287227 sents, 17123886 tokens, 0.0% replaced by <unk>
| [tgt] Dictionary: 30391 types
| [tgt] examples/summarization/fairseq_data/trunc400/valid.tgt: 13368 sents, 886214 tokens, 0.0% replaced by <unk>
| [tgt] Dictionary: 30391 types
| [tgt] examples/summarization/fairseq_data/trunc400/test.tgt: 11490 sents, 726123 tokens, 0.000275% replaced by <unk>
```

## Train
```
nvidia-docker run --rm -it -v $(pwd):/workspace --ipc=host pytorch/pytorch
```

### [Official transformer](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)

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

**Big transformer**
```
# Local
arch=transformer_vaswani_wmt_en_de_big
mkdir -p checkpoints/$arch
python train.py data-bin/cnndm -s src -t tgt --max-tokens 4000 -a $arch --share-all-embeddings --dropout 0.3 --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --weight-decay 0.0001 --lr 0.0005 --warmup-init-lr 1e-07 --warmup-updates 4000 --lr-scheduler inverse_sqrt --min-lr 1e-09 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-update 300000 --save-dir checkpoints/$arch
python scripts/average_checkpoints.py --inputs checkpoints/$arch --num-epoch-checkpoints 20 --output checkpoints/$arch/model.pt
python generate.py data-bin/cnndm --path checkpoints/$arch/model.pt --batch-size 128 --beam 5 --remove-bpe --no-repeat-ngram-size 3

# Philly environmentVariables 
rootdir: "/philly/eu2/ipgsrch/yushi/fairseq",
arch: "transformer_vaswani_wmt_en_de_big",
datadir: "data-bin/cnndm"

# Philly commandLine
python $rootdir/train.py $rootdir/$datadir -s src -t tgt --max-tokens 4000 -a $arch --share-all-embeddings --dropout 0.3 --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --weight-decay 0.0001 --lr 0.0005 --warmup-init-lr 1e-07 --warmup-updates 4000 --lr-scheduler inverse_sqrt --min-lr 1e-09 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-update 300000 --save-dir $PHILLY_JOB_DIRECTORY
python $rootdir/generate.py $rootdir/$datadir --path ${modelpath}_1555486458178_5362/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe --no-repeat-ngram-size 3
```


### [Base transformer for summarization](https://arxiv.org/pdf/1904.01038.pdf)

> We truncate articles to `400` tokens (See et al., 2017). We use BPE with `30K` operations to form our vocabulary following Fan et al. (2018a). To evaluate, we use the standard ROUGE metric (Lin, 2004) and report ROUGE-1, ROUGE-2, and ROUGE-L. To generate summaries, we follow standard practice in `tuning the minimum output length` and `disallow repeating the same trigram` (Paulus et al., 2017).

> We also consider a configuration where we input `pre-trained` language model representations to the encoder network and this
language model was trained on `newscrawl and CNN-Dailymail`, totalling `193M` sentences.

```
arch=transformer
mkdir -p checkpoints/$arch
python train.py data-bin/cnndm.400 -a $arch --optimizer adam --lr 0.0005 -s src -t tgt --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 --min-lr 1e-09 --lr-scheduler inverse_sqrt --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --max-update 50000 --warmup-updates 4000 --warmup-init-lr 1e-07 --adam-betas '(0.9, 0.98)' --save-dir checkpoints/$arch
python scripts/average_checkpoints.py --inputs checkpoints/$arch --num-epoch-checkpoints 10 --output checkpoints/$arch/model.pt
python generate.py data-bin/cnndm.400 --path checkpoints/$arch/model.pt --batch-size 128 --beam 5 --remove-bpe --no-repeat-ngram-size 3
```

### [Lightweight convolutions](https://openreview.net/pdf?id=SkVhlh09tX)

> We test the model’s ability to process long documents on the CNN-DailyMail summarization task (Hermann et al., 2015; Nallapati et al., 2016) comprising over 280K news articles paired with multi-sentence summaries. Articles are truncated to `400 tokens` (See et al., 2017) and we use a BPE vocabulary of `30K types` (Fan et al., 2017). We evaluate in terms of F1-Rouge, that is Rouge-1, Rouge-2 and Rouge-L (Lin, 2004) (ROUGE-1.5.5.pl parameters: `-m -a -n 2`). When generating summaries, we follow standard practice in `tuning the maximum output length`, `disallowing repeating the same trigram`, and we apply a `stepwise length penalty` (Paulus et al., 2017; Fan et al., 2017; Wu et al., 2016).

> We train with `Adam` using the `cosine` learning rate schedule with a warmup of
`10K` steps and a period of `20K` updates. We use weight decay `1e-3` and dropout `0.3`.

> We reduce model capacity by setting `d = 1024`, `dff = 2048`, `H = 8`, similar to the Transformer base setup of Vaswani et al. (2017).

```
arch=lightconv
mkdir -p checkpoints/$arch
python train.py data-bin/cnndm.400 -s src -t tgt -a $arch --share-all-embeddings --update-freq 16 --clip-norm 0.0 --optimizer adam --lr 1e-7 --max-tokens 4000 --no-progress-bar --log-interval 100 --min-lr 1e-09 --weight-decay 1e-3 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --ddp-backend=no_c10d --max-update 50000 --adam-betas '(0.9, 0.98)' --lr-scheduler cosine --warmup-updates 10000 --warmup-init-lr 1e-7 --lr-period-updates 20000 --lr-shrink 1 --max-lr 0.001 --t-mult 1 --dropout 0.3 --attention-dropout 0.1 --weight-dropout 0.1 --keep-last-epochs 10 --encoder-glu 1 --decoder-glu 1 --save-dir checkpoints/$arch
```

### transformer_iwslt_de_en

```
arch=transformer_iwslt_de_en
mkdir -p checkpoints/$arch
python train.py data-bin/cnndm.400 -a $arch --optimizer adam --lr 0.0005 -s src -t tgt --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 --min-lr 1e-09 --lr-scheduler inverse_sqrt --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --max-update 50000 --warmup-updates 4000 --warmup-init-lr 1e-07 --adam-betas '(0.9, 0.98)' --save-dir checkpoints/$arch
python scripts/average_checkpoints.py --inputs checkpoints/$arch --num-epoch-checkpoints 10 --output checkpoints/$arch/model.pt
python generate.py data-bin/cnndm.400 --path checkpoints/$arch/model.pt --batch-size 128 --beam 5 --remove-bpe
```

### Scaling NMT

```
arch=transformer_vaswani_wmt_en_de_big
mkdir -p checkpoints/$arch
python train.py data-bin/cnndm.400 --arch $arch --share-all-embeddings --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0005 --min-lr 1e-09 --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 3584 --fp16 --update-freq 1 --save-dir checkpoints/$arch
```

### Mixture models
```
arch=transformer_vaswani_wmt_en_de
mkdir -p checkpoints/$arch
python train.py data-bin/cnndm.400 --max-update 100000 --task translation_moe --method hMoElp --mean-pool-gating-network --num-experts 3 --arch $arch --share-all-embeddings --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0007 --min-lr 1e-09 --dropout 0.1 --weight-decay 0.0 --criterion cross_entropy --max-tokens 3584 --update-freq 8
python generate.py data-bin/cnndm.400 --path checkpoints/$arch/checkpoint_best.pt --beam 1 --remove-bpe --task translation_moe --method hMoElp --mean-pool-gating-network --num-experts 3 --gen-expert 0
```

# Evaluation

## Install pyrouge
```
git clone https://github.com/andersjo/pyrouge
cd pyrouge/tools/ROUGE-1.5.5/data
rm WordNet-2.0.exc.db
./WordNet-2.0-Exceptions/buildExeptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db
cd ../../../..
apt-get update && apt-get install -y perl synaptic
pip install pyrouge
```

## Download result
```
id=1555486458178_7344
sudo bash /home/yushi/philly-fs.bash -cp //philly/eu2/ipgsrch/sys/jobs/application_$id/cnndm_step50000.candidate results/
sudo bash /home/yushi/philly-fs.bash -cp //philly/eu2/ipgsrch/sys/jobs/application_$id/cnndm_step50000.gold results/
docker run --rm -it -v $(pwd):/workspace bertsum
pyrouge_set_rouge_path pyrouge/tools/ROUGE-1.5.5
python src/rouge.py
```
