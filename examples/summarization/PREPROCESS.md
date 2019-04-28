# Data processing

This need be run in folder examples/summarization

## Generate raw data by following https://github.com/shi2yu3/BertSum/PROCESS.md, then copy data to current folder
```
mkdir fairseq_data/
cp -r BertSum/fairseq_data/trunc400/ fairseq_data/
```

## Download BPE tool

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

## Binarize the dataset

This need be run in the root folder

```
cd ../../
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
