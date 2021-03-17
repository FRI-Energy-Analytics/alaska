"""
Tests for alaska/model.py
"""
import torch.nn as nn
from torch import optim
from ..model import Seq2Seq, DEVICE
from ..params import Params
from ..utils import Dataset, Vocab


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
    vocab = dataset.build_vocab(params.vocab_size, embed_file=params.embed_file)
    model = Seq2Seq(vocab, params)

    assert model.state_dict()["enc_dec_adapter.weight"].size()[0] == 200
    assert model.state_dict()["enc_dec_adapter.weight"].size()[1] == 300
    assert model.state_dict()["enc_dec_adapter.bias"].size()[0] == 200
    assert model.state_dict()["embedding.weight"].size()[0] == 859
    assert model.state_dict()["embedding.weight"].size()[1] == 100


def test_model_eval():
    """
    Test the forward pass through the model
    """
    params = Params()
    dataset = Dataset(
        params.data_path,
        max_src_len=params.max_src_len,
        max_tgt_len=params.max_tgt_len,
        truncate_src=params.truncate_src,
        truncate_tgt=params.truncate_tgt,
    )
    # build the vocab
    vocab = dataset.build_vocab(params.vocab_size, embed_file=params.embed_file)
    # instantiate the model
    model = Seq2Seq(vocab, params)
    model.eval()
    # create the data generator
    train_gen = dataset.generator(
        params.batch_size, vocab, vocab, True if params.pointer else False
    )
    batch = next(train_gen)
    # set the model parameters
    vocab = Vocab
    criterion = nn.NLLLoss(ignore_index=vocab.PAD)
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    forcing_ratio = params.forcing_ratio
    partial_forcing = params.partial_forcing
    sample = params.sample
    show_cover_loss = params.show_cover_loss
    ext_vocab_size = batch.ext_vocab_size
    input_lengths = batch.input_lengths
    # zero the optimizer gradient and send to device
    optimizer.zero_grad()
    input_tensor = batch.input_tensor.to(DEVICE)
    target_tensor = batch.target_tensor.to(DEVICE)
    # run a forward pass through the seq2seq model
    forward_pass = model(
        input_tensor,
        target_tensor,
        input_lengths,
        criterion,
        forcing_ratio=forcing_ratio,
        partial_forcing=partial_forcing,
        sample=sample,
        ext_vocab_size=ext_vocab_size,
        include_cover_loss=show_cover_loss,
    )

    assert forward_pass.loss.item() > 12
    assert forward_pass.loss_value > 10
    assert list(forward_pass.encoder_hidden.shape) == [1, 32, 300]
