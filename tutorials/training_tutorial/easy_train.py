"""
Simple training using pointer generator, code modified from ymfa/seq2seq-summarizer 
https://github.com/ymfa/seq2seq-summarizer/blob/master/
"""
from params import Params
from train import train
from utils import Dataset
from model import Seq2Seq
import torch

p = Params() # initialize parameters 
data_path = ""  # path to training data, must be .gz file
val_data_path = ""  # path to validation data, must be .gz file
state_dict_path = "" # path to saving state dictionary

# build dataset
dataset = Dataset(
    data_path,
    max_src_len=p.max_src_len,
    max_tgt_len=p.max_tgt_len,
    truncate_src=p.truncate_src,
    truncate_tgt=p.truncate_tgt,
)

# build vocabulary
v = dataset.build_vocab(p.vocab_size, embed_file=p.embed_file)
m = Seq2Seq(v, p)

train_gen = dataset.generator(p.batch_size, v, v, True if p.pointer else False)
# validation dataset
if val_data_path:
    val_dataset = Dataset(
        val_data_path,
        max_src_len=p.max_src_len,
        max_tgt_len=p.max_tgt_len,
        truncate_src=p.truncate_src,
        truncate_tgt=p.truncate_tgt,
    )
    val_gen = val_dataset.generator(
        p.val_batch_size, v, v, True if p.pointer else False
    )
else:
    val_gen = None

train(train_gen, v, m, p, val_gen)
# print sizes of layers in state dictionary
print("Model's state_dict:")
for param_tensor in m.state_dict():
    print(param_tensor, "\t", m.state_dict()[param_tensor].size())
# save model to path
torch.save(m.state_dict(), state_dict_path)
