"""
Tests for alaska/model.py
"""
import torch
from pathlib import Path
from ..model import Seq2Seq
from ..params import Params
from ..utils import Dataset, Vocab
# not sure where to import train from
from ..train import train

def test_forward():
    """
    Test that we can instantiate the Seq2Seq class
    """
    params = Params()
    dataset = Dataset(
        params.data_path,
        max_src_len=params.max_src_len,
        max_tgt_len=params.max_tgt_len,
        truncate_src=params.truncate_src,
        truncate_tgt=params.truncate_tgt,
    )
    vc = dataset.build_vocab(params.vocab_size, embed_file=params.embed_file)
    m = Seq2Seq(vc, params)

    train_gen = dataset.generator(params.batch_size, vc, vc, True)

    train(train_gen, vc, m, params, None)
    assert m.state_dict()["enc_dec_adapter.weight"].size()[0] \
        == 200
    assert m.state_dict()["enc_dec_adapter.weight"].size()[1] \
        == 300
    assert m.state_dict()["enc_dec_adapter.bias"].size()[0] \
        == 200
    assert m.state_dict()["embedding.weight"].size()[0] \
        == 858
    assert m.state_dict()["embedding.weight"].size()[1] \
        == 100

