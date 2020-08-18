# validate state dict, TO BE DELETED
import torch
import math
from utils import Vocab, OOVDict, Batch, format_tokens, Dataset
from model import DEVICE, Seq2SeqOutput, Seq2Seq
from params import Params

def print_dict():
    p = Params()
    dataset = Dataset(
        p.data_path,
        max_src_len=p.max_src_len,
        max_tgt_len=p.max_tgt_len,
        truncate_src=p.truncate_src,
        truncate_tgt=p.truncate_tgt,
    )
    v = dataset.build_vocab(p.vocab_size, embed_file=p.embed_file)
    m = Seq2Seq(v, p)
    m.load_state_dict(torch.load("state_dict.pth"))
    m.eval()
    print("state dict")
    print(m.state_dict())

print_dict()