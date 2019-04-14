# CNN/DailyMail leaderboard

## Non-annonymized

Modified from http://nlpprogress.com/english/summarization.html

| Model           | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR | Paper / Source | Code |
| --------------- | :-----: | :-----: | :-----: | :----: | -------------- | ---- |
| BERTSUM+Transformer (Liu, 2019) | 43.25 | 20.24 | 39.63 | - | [Fine-tune BERT for Extractive Summarization](https://arxiv.org/pdf/1903.10318.pdf) | [Official](https://github.com/nlpyang/BertSum) |
| EditNet (Moroshko et al., 2019) | 41.42 | 19.03 | 38.36 | - | [An Editorial Network For Enhanced Document Summarization](https://arxiv.org/pdf/1902.10360.pdf) | |
| DCA (Celikyilmaz et al., 2018) | 41.69 | 19.47 | 37.92 | - | [Deep Communicating Agents for Abstractive Summarization](http://aclweb.org/anthology/N18-1150) | |
| NeuSUM (Zhou et al., 2018) | 41.59 | 19.01 | 37.98 | - | [Neural Document Summarization by Jointly Learning to Score and Select Sentences](http://aclweb.org/anthology/P18-1061) | [Official](https://github.com/magic282/NeuSum) |
| Latent (Zhang et al., 2018) | 41.05 | 18.77 | 37.54 | - | [Neural Latent Extractive Document Summarization](http://aclweb.org/anthology/D18-1088) | | 
| rnn-ext + RL (Chen and Bansal, 2018) | 41.47 | 18.72 | 37.76 | 22.35 | [Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting](http://aclweb.org/anthology/P18-1061) | [Official](https://github.com/chenrocks/fast_abs_rl) |
| BanditSum (Dong et al., 2018) | 41.5 | 18.7 | 37.6 | - | [BANDITSUM: Extractive Summarization as a Contextual Bandit](http://aclweb.org/anthology/D18-1208) | |
| Bottom-Up Summarization (Gehrmann et al., 2018) | 41.22 | 18.68 | 38.34 | - | [Bottom-Up Abstractive Summarization](https://arxiv.org/abs/1808.10792) | [Official](https://github.com/sebastianGehrmann/bottom-up-summary) |
| REFRESH (Narayan et al., 2018) | 40.0 | 18.2 | 36.6 | - | [Ranking Sentences for Extractive Summarization with Reinforcement Learning](http://aclweb.org/anthology/N18-1158) | [Official](https://github.com/EdinburghNLP/Refresh) |
| (Li et al., 2018a) | 41.54 | 18.18 | 36.47 | - | [Improving Neural Abstractive Document Summarization with Explicit Information Selection Modeling](http://aclweb.org/anthology/D18-1205) | |
| (Li et al., 2018b) | 40.30 | 18.02 | 37.36 | - | [Improving Neural Abstractive Document Summarization with Structural Regularization](http://aclweb.org/anthology/D18-1441) | |
| ROUGESal+Ent RL (Pasunuru and Bansal, 2018) | 40.43 | 18.00 | 37.10 | 20.02 | [Multi-Reward Reinforced Summarization with Saliency and Entailment](http://aclweb.org/anthology/N18-2102) | |
| end2end w/ inconsistency loss (Hsu et al., 2018) | 40.68 | 17.97 | 37.13 | - | [A Unified Model for Extractive and Abstractive Summarization using Inconsistency Loss](http://aclweb.org/anthology/P18-1013) | |
| RL + pg + cbdec (Jiang and Bansal, 2018) | 40.66 | 17.87 | 37.06 | 20.51 | [Closed-Book Training to Improve Summarization Encoder Memory](http://aclweb.org/anthology/D18-1440) | |
| rnn-ext + abs + RL + rerank (Chen and Bansal, 2018) | 40.88 | 17.80 | 38.54 | 20.38 | [Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting](http://aclweb.org/anthology/P18-1061) | [Official](https://github.com/chenrocks/fast_abs_rl) |
| Lead-3 baseline (See et al., 2017) | 40.34 | 17.70 | 36.57 | 22.21 | [Get To The Point: Summarization with Pointer-Generator Networks](http://aclweb.org/anthology/P17-1099) | [Official](https://github.com/abisee/pointer-generator) |
| Pointer + Coverage + EntailmentGen + QuestionGen (Guo et al., 2018) | 39.81 | 17.64 | 36.54 | 18.54 | [Soft Layer-Specific Multi-Task Summarization with Entailment and Question Generation](http://aclweb.org/anthology/P18-1064) | |
| ML+RL ROUGE+Novel, with LM (Kryscinski et al., 2018) | 40.19 | 17.38 | 37.52 | - | [Improving Abstraction in Text Summarization](http://aclweb.org/anthology/D18-1207) | |
| Pointer-generator + coverage (See et al., 2017) | 39.53 | 17.28 | 36.38 | 18.72 | [Get To The Point: Summarization with Pointer-Generator Networks](http://aclweb.org/anthology/P17-1099) | [Official](https://github.com/abisee/pointer-generator) |


# Paper list

```
https://github.com/mathsyouth/awesome-text-summarization
```


# Data Preparation For CNN/Dailymail

## Step 1 Download Stories
Download and unzip the `stories` directories from [here](http://cs.nyu.edu/~kcho/DMQA/) for both CNN and Daily Mail. Put all  `.stroy` files in one directory (e.g. `../raw_stories`)

####  Step 2. Download Stanford CoreNLP
We will need Stanford CoreNLP to tokenize the data. Download it [here](https://stanfordnlp.github.io/CoreNLP/) and unzip it. Then add the following command to your bash_profile:
```
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
unzip stanford-corenlp-full-2018-10-05.zip
export CLASSPATH=/path/to/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar
```
replacing `/path/to/` with the path to where you saved the `stanford-corenlp-full-2018-10-05` directory.

####  Step 3. Sentence Splitting and Tokenization

```
python preprocess.py -mode tokenize -raw_path RAW_PATH -save_path TOKENIZED_PATH
git clone https://github.com/nlpyang/BertSum
python BertSum/src/preprocess.py -mode tokenize -raw_path cnn -save_path cnn_tok
```

* `RAW_PATH` is the directory containing story files (`../raw_stories`), `JSON_PATH` is the target directory to save the generated json files (`../merged_stories_tokenized`)


####  Step 4. Format to Simpler Json Files
 
```
python preprocess.py -mode format_to_lines -raw_path RAW_PATH -save_path JSON_PATH -map_path MAP_PATH -lower 
```

* `RAW_PATH` is the directory containing tokenized files (`../merged_stories_tokenized`), `JSON_PATH` is the target directory to save the generated json files (`../json_data/cnndm`), `MAP_PATH` is the  directory containing the urls files (`../urls`)

####  Step 5. Format to PyTorch Files
```
python preprocess.py -mode format_to_bert -raw_path JSON_PATH -save_path BERT_DATA_PATH -oracle_mode greedy -n_cpus 4 -log_file ../logs/preprocess.log
```

* `JSON_PATH` is the directory containing json files (`../json_data`), `BERT_DATA_PATH` is the target directory to save the generated binary files (`../bert_data`)

* `-oracle_mode` can be `greedy` or `combination`, where `combination` is more accurate but takes much longer time to process 

