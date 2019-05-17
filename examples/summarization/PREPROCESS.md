# Data processing

This need be run in folder examples/summarization
```
cd examples/summarization
```

## Get raw data

### **Option 1**: following https://github.com/shi2yu3/BertSum/PROCESS.md, then copy to fairseq_data/
```
mkdir -p data/raw/bertsum/trunc400
cp ~/BertSum/fairseq_data/trunc400/* data/raw/bertsum/trunc400
type=bertsum
data_path=data/raw/bertsum/trunc400
src_ext=src.txt
tgt_ext=tgt.txt
```

### **Option 2**: following https://github.com/shi2yu3/opennmt/SUMMARIZATION.md
```
mkdir -p data/raw/opennmt
cd data/raw/opennmt
curl -o cnndm.tar.gz https://s3.amazonaws.com/opennmt-models/Summary/cnndm.tar.gz
tar xzf cnndm.tar.gz
rm cnndm.tar.gz
cd ../../..
type=opennmt
data_path=data/raw/opennmt
src_ext=txt.src
tgt_ext=txt.tgt.tagged
```

## Download BPE tool

```
git clone https://github.com/rsennrich/subword-nmt.git
```

## Generate BPE code
```
cat $data_path/train.$src_ext $data_path/train.$tgt_ext > $data_path/train.txt
python subword-nmt/learn_bpe.py -s 30000 < $data_path/train.txt > $data_path/code
```

## Truncation
```
python ../../truncate.py $data_path/train.$src_ext 400
python ../../truncate.py $data_path/train.$tgt_ext 100
python ../../truncate.py $data_path/test.$src_ext 400
python ../../truncate.py $data_path/test.$tgt_ext 100
python ../../truncate.py $data_path/val.$src_ext 400
python ../../truncate.py $data_path/val.$tgt_ext 100
src_ext=txt.src.400
tgt_ext=txt.tgt.tagged.100
```

## Tokenization
```
python subword-nmt/apply_bpe.py -c $data_path/code < $data_path/train.$src_ext > $data_path/train.src
python subword-nmt/apply_bpe.py -c $data_path/code < $data_path/train.$tgt_ext > $data_path/train.tgt
python subword-nmt/apply_bpe.py -c $data_path/code < $data_path/test.$src_ext > $data_path/test.src
python subword-nmt/apply_bpe.py -c $data_path/code < $data_path/test.$tgt_ext > $data_path/test.tgt
python subword-nmt/apply_bpe.py -c $data_path/code < $data_path/val.$src_ext > $data_path/valid.src
python subword-nmt/apply_bpe.py -c $data_path/code < $data_path/val.$tgt_ext > $data_path/valid.tgt
```

## Binarize the dataset

This need be run in the root folder

```
cd ../../
docker run --rm -it -v $(pwd):/workspace bertsum
python preprocess.py --source-lang src --target-lang tgt --joined-dictionary --trainpref examples/summarization/$data_path/train --validpref examples/summarization/$data_path/valid --testpref examples/summarization/$data_path/test --destdir data-bin/cnndm/$type/
```
Output using BertSum
```
| [src] Dictionary: 30391 types
| [src] examples/summarization/$data_path/train.src: 287227 sents, 118030308 tokens, 0.0% replaced by <unk>
| [src] Dictionary: 30391 types
| [src] examples/summarization/$data_path/valid.src: 13368 sents, 5451317 tokens, 0.000147% replaced by <unk>
| [src] Dictionary: 30391 types
| [src] examples/summarization/$data_path/test.src: 11490 sents, 4700782 tokens, 0.000425% replaced by <unk>
| [tgt] Dictionary: 30391 types
| [tgt] examples/summarization/$data_path/train.tgt: 287227 sents, 17123886 tokens, 0.0% replaced by <unk>
| [tgt] Dictionary: 30391 types
| [tgt] examples/summarization/$data_path/valid.tgt: 13368 sents, 886214 tokens, 0.0% replaced by <unk>
| [tgt] Dictionary: 30391 types
| [tgt] examples/summarization/$data_path/test.tgt: 11490 sents, 726123 tokens, 0.000275% replaced by <unk>
```
Output using OpenNMT
```
| [src] Dictionary: 30415 types
| [src] examples/summarization/data/raw/opennmt/train.src: 287227 sents, 118100031 tokens, 0.0% replaced by <unk>
| [src] Dictionary: 30415 types
| [src] examples/summarization/data/raw/opennmt/valid.src: 13368 sents, 5454970 tokens, 0.00011% replaced by <unk>
| [src] Dictionary: 30415 types
| [src] examples/summarization/data/raw/opennmt/test.src: 11490 sents, 4704656 tokens, 0.000446% replaced by <unk>
| [tgt] Dictionary: 30415 types
| [tgt] examples/summarization/data/raw/opennmt/train.tgt: 287227 sents, 18877554 tokens, 0.0% replaced by <unk>
| [tgt] Dictionary: 30415 types
| [tgt] examples/summarization/data/raw/opennmt/valid.tgt: 13368 sents, 962775 tokens, 0.0% replaced by <unk>
| [tgt] Dictionary: 30415 types
| [tgt] examples/summarization/data/raw/opennmt/test.tgt: 11490 sents, 792882 tokens, 0.000126% replaced by <unk>
| Wrote preprocessed data to data-bin/cnndm/opennmt
```



# Mapping between FairSeq and OpenNMT

**OpenNMT**
```
python preprocess.py -train_src data/cnndm/train.txt.src -train_tgt data/cnndm/train.txt.tgt.tagged -valid_src data/cnndm/val.txt.src -valid_tgt data/cnndm/val.txt.tgt.tagged -save_data data/cnndm/CNNDM -src_seq_length 10000 -tgt_seq_length 10000 -src_seq_length_trunc 400 -tgt_seq_length_trunc 100 -dynamic_dict -share_vocab -shard_size 100000

python train.py --save_model models/cnndm --data data/cnndm/CNNDM --copy_attn --global_attention mlp --word_vec_size 128 --rnn_size 512 --layers 1 --encoder_type brnn --train_steps 200000 --max_grad_norm 2 --dropout 0. --batch_size 16 --valid_batch_size 16 --optim adagrad --learning_rate 0.15 --adagrad_accumulator_init 0.1 --reuse_copy_attn --copy_loss_by_seqlength --bridge --seed 777 --world_size 2 --gpu_ranks 0 1
```

**FairSeq**
```
python preprocess.py --source-lang src --target-lang tgt --joined-dictionary --trainpref examples/summarization/$data_path/train --validpref examples/summarization/$data_path/valid --testpref examples/summarization/$data_path/test --destdir data-bin/cnndm

python train.py data-bin/cnndm -s src -t tgt --max-tokens 4000 -a transformer_vaswani_wmt_en_de_big --share-all-embeddings --dropout 0.3 --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --weight-decay 0.0001 --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --min-lr 1e-9 --warmup-updates 4000 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-update 350000 --max-epoch 10 --save-dir checkpoints --tensorboard-logdir tensorboard
```

**FairSeq for OpenNMT**
```
python train.py data/cnndm/CNNDM --save-dir checkpoints/cnndm --task summarization -a opennmt --copy_attn --global_attention mlp --word_vec_size 128 --rnn_size 512 --layers 1 --encoder_type brnn --max-update 200000 --clip-norm 2 --dropout 0. --max-sentences 16 --max-sentences-valid 16 --optimizer adagrad --lr 0.15 --adagrad_accumulator_init 0.1 --reuse_copy_attn --copy_loss_by_seqlength --bridge --seed 777 --criterion copy_generator_loss
```
