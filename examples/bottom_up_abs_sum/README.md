# Train

python train.py data/cnndm -a opennmt_cnndm --copy_attn --global_attention mlp --word_vec_size 128 --rnn_size 512 --layers 1 --encoder_type brnn --train_steps 200000 --max_grad_norm 2 --dropout 0. --batch_size 16 --valid_batch_size 16 --optimizer adagrad --learning_rate 0.15 --adagrad_accumulator_init 0.1 --reuse_copy_attn --copy_loss_by_seqlength --bridge --seed 777 --world_size 2 --gpu_ranks 0 1 --save-dir checkpoints

# Reference

http://opennmt.net/OpenNMT-py/Summarization.html

```
@inproceedings{gehrmann2018bottom,
  title={Bottom-Up Abstractive Summarization},
  author={Gehrmann, Sebastian and Deng, Yuntian and Rush, Alexander},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  pages={4098--4109},
  year={2018}
}
```
