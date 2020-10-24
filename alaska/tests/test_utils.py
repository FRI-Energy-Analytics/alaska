"""
Tests for alaska/utils.py
"""
from pathlib import Path
from ..params import Params
from ..utils import Dataset, Vocab


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
