# MIT License
# Copyright (c) 2018 Yimai Fang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Code modified from Yimai Fang's seq2seq-summarizer
# repository: https://github.com/ymfa/seq2seq-summarizer

# Copyright (c) 2021 The AlasKA Developers.
# Distributed under the terms of the MIT License.
# SPDX-License_Identifier: MIT
"""
Tests for alaska/utils.py
"""
from pathlib import Path
import matplotlib.pyplot as plt
from ..params import Params
from ..utils import Dataset, Vocab, show_plot, show_attention_map


def test_vocab_add_words():
    """
    Test that we can instantiate the Vocab class and add words
    """
    vc = Vocab()
    vc.add_words(["mwd", "lwd", "lwd"])
    assert vc.word2index == {"mwd": 4, "lwd": 5}
    assert vc.word2count["lwd"] == 2
    assert vc.index2word == ["<PAD>", "<SOS>", "<EOS>", "<UNK>", "mwd", "lwd"]


def test_vocab_trim_by_frequency():
    """
    Test that the vocab class trim method works with min frequency of 2
    """
    vc = Vocab()
    vc.add_words(["mwd", "lwd", "lwd"])
    vc.trim(min_freq=2, vocab_size=None)
    assert vc.word2index == {"lwd": 4}
    assert vc.word2count["lwd"] == 2
    assert vc.index2word == ["<PAD>", "<SOS>", "<EOS>", "<UNK>", "lwd"]


def test_vocab_trim_by_size():
    """
    Test that the vocab class trim method works with min vocab size of 2
    """
    vc = Vocab()
    vc.add_words(["mwd", "lwd", "lwd"])
    vc.trim(min_freq=1, vocab_size=2)
    assert vc.word2index == {"mwd": 4, "lwd": 5}
    assert vc.word2count["lwd"] == 2
    assert vc.index2word == ["<PAD>", "<SOS>", "<EOS>", "<UNK>", "mwd", "lwd"]


def test_vocab_embeddings():
    """
    Test that the vocab class embeddings are equal to None at instantiation
    """
    vc = Vocab()
    assert vc.embeddings is None


def test_vocab_getitem():
    """
    Test that the vocab class can get items
    """
    vc = Vocab()
    vc.add_words(["mwd", "lwd", "lwd"])
    assert vc.__getitem__(5) == "lwd"
    assert vc.__getitem__("lwd") == 5


def test_vocab_len():
    """
    Test that the vocab class can return its length
    """
    vc = Vocab()
    vc.add_words(["mwd", "lwd", "lwd"])
    assert vc.__len__() == 6


def test_vocab_words():
    """
    Test that the vocab class can return words and words with punctuation
    """
    vc = Vocab()
    #             [4]    [5]           [6]     [7]  [8]
    vc.add_words(["mwd", "lwd", "lwd", "lwd.", ".", "<P>"])

    # token_id < 4 is False
    assert vc.is_word(token_id=3) is not True
    assert vc.is_word(token_id=4) is True

    # token_id >= Vocab.__len__()
    assert vc.is_word(token_id=9) is True
    assert vc.is_word(token_id=10) is True

    # token_str is a word or not a word
    assert vc.is_word(token_id=5) is True
    assert vc.is_word(token_id=6) is True
    assert vc.is_word(token_id=7) is False
    assert vc.is_word(token_id=8) is False


def test_dataset_1():
    """
    Test Dataset object
    """
    params = Params()
    dataset = Dataset(
        # "alaska/data/mnem.gz"
        params.data_path,
        max_src_len=params.max_src_len,
        max_tgt_len=params.max_tgt_len,
        truncate_src=params.truncate_src,
        truncate_tgt=params.truncate_tgt,
    )
    assert dataset.src_len == 17
    assert dataset.tgt_len == 5
    assert len(dataset.pairs) == 1237
    assert "caliper" in dataset.pairs[0].tgt
    assert dataset.pairs[0].src == ["caliper", "1", "in", "c1"]


def test_dataset_2():
    """
    Test Dataset object: build_vocab without input file
    """
    params = Params()
    dataset = Dataset(
        # "alaska/data/mnem.gz"
        params.data_path,
        max_src_len=params.max_src_len,
        max_tgt_len=params.max_tgt_len,
        truncate_src=params.truncate_src,
        truncate_tgt=params.truncate_tgt,
    )
    # Example standard call to build_vocab()
    # vocab = dataset.build_vocab(
    # 30000,
    # embed_file="alaska/data/.vector_cache/glove.6B.100d.txt)

    # Call without vocab_size creates a non-existant filename
    vocab = dataset.build_vocab(embed_file=params.embed_file)
    assert len(vocab) == 858

    # Clean up generated .vocab file created by build_vocab
    filename = Path(dataset.filename).with_suffix("").with_suffix(".vocab")
    try:
        filename.unlink()
    except FileNotFoundError:
        pass


def test_show_plot():
    """
    Test show_plot function can save to disk
    """
    loss = [10, 9, 8]
    step = 1
    val_loss = [1, 2, 3]
    val_metric = [3, 2, 1]
    val_step = 1
    file_prefix = "show_plot_test"
    show_plot(
        loss=loss,
        step=step,
        val_loss=val_loss,
        val_metric=val_metric,
        val_step=val_step,
        file_prefix=file_prefix,
    )
    filename = Path(file_prefix).with_suffix("").with_suffix(".png")
    assert str(filename) == "show_plot_test.png"
    try:
        filename.unlink()
    except FileNotFoundError:
        pass


def test_show_attention_map():
    """
    Test show_attention_map function can create a plot without failure
    """
    source = ["environmentally", "corrected", "gamma", "ray"]
    preds = ["gamma", "ray"]
    attention = [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 4, 0], [0, 0, 0, 5]]
    point_ratio = [1, 2, 3]
    show_attention_map(
        src_words=source,
        pred_words=preds,
        attention=attention,
        pointer_ratio=point_ratio,
    )
    plt.gcf().canvas.draw()
