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
General data processing and tokenizing functions
Some plotting functions as well as functions for calulating rouge scores
"""
import os
import re
from tempfile import TemporaryDirectory
import subprocess
from multiprocessing.dummy import Pool
import gzip
from functools import lru_cache
from random import shuffle
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import NamedTuple, List, Callable, Dict, Tuple, Optional
import torch


plt.switch_backend("agg")

word_detector = re.compile(r"\w")


class Vocab(object):

    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 3

    def __init__(self):
        self.word2index = {}
        self.word2count = Counter()
        self.reserved = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
        self.index2word = self.reserved[:]
        self.embeddings = None

    def add_words(self, words: List[str]):
        for word in words:
            if word not in self.word2index:
                self.word2index[word] = len(self.index2word)
                self.index2word.append(word)
        self.word2count.update(words)

    def trim(self, *, vocab_size: int = None, min_freq: int = 1):
        if min_freq <= 1 and (vocab_size is None or vocab_size >= len(self.word2index)):
            return
        ordered_words = sorted(
            ((c, w) for (w, c) in self.word2count.items()), reverse=True
        )
        if vocab_size:
            ordered_words = ordered_words[:vocab_size]
        self.word2index = {}
        self.word2count = Counter()
        self.index2word = self.reserved[:]
        for count, word in ordered_words:
            if count < min_freq:
                break
            self.word2index[word] = len(self.index2word)
            self.word2count[word] = count
            self.index2word.append(word)

    def load_embeddings(self, file_path: str, dtype=np.float32) -> int:
        num_embeddings = 0
        vocab_size = len(self)
        with open(file_path, "rb") as f:
            for line in f:
                line = line.split()
                word = line[0].decode("utf-8")
                idx = self.word2index.get(word)
                if idx is not None:
                    vec = np.array(line[1:], dtype=dtype)
                    if self.embeddings is None:
                        n_dims = len(vec)
                        self.embeddings = np.random.normal(
                            np.zeros((vocab_size, n_dims))
                        ).astype(dtype)
                        self.embeddings[self.PAD] = np.zeros(n_dims)
                    self.embeddings[idx] = vec
                    num_embeddings += 1
        return num_embeddings

    def __getitem__(self, item):
        if type(item) is int:
            return self.index2word[item]
        return self.word2index.get(item, self.UNK)

    def __len__(self):
        return len(self.index2word)

    @lru_cache(maxsize=None)
    def is_word(self, token_id: int) -> bool:
        """Return whether the token at `token_id` is a word; False for punctuations."""
        if token_id < 4:
            return False
        if token_id >= len(self):
            return True  # OOV is assumed to be words
        token_str = self.index2word[token_id]
        if not word_detector.search(token_str) or token_str == "<P>":
            return False
        return True


class Example(NamedTuple):
    src: List[str]
    tgt: List[str]
    src_len: int  # inclusive of EOS, so that it corresponds to tensor shape
    tgt_len: int  # inclusive of EOS, so that it corresponds to tensor shape


class OOVDict(object):
    def __init__(self, base_oov_idx):
        self.word2index = {}  # type: Dict[Tuple[int, str], int]
        self.index2word = {}  # type: Dict[Tuple[int, int], str]
        self.next_index = {}  # type: Dict[int, int]
        self.base_oov_idx = base_oov_idx
        self.ext_vocab_size = base_oov_idx

    def add_word(self, idx_in_batch, word) -> int:
        key = (idx_in_batch, word)
        index = self.word2index.get(key)
        if index is not None:
            return index
        index = self.next_index.get(idx_in_batch, self.base_oov_idx)
        self.next_index[idx_in_batch] = index + 1
        self.word2index[key] = index
        self.index2word[(idx_in_batch, index)] = word
        self.ext_vocab_size = max(self.ext_vocab_size, index + 1)
        return index


class Batch(NamedTuple):
    examples: List[Example]
    input_tensor: Optional[torch.Tensor]
    target_tensor: Optional[torch.Tensor]
    input_lengths: Optional[List[int]]
    oov_dict: Optional[OOVDict]

    @property
    def ext_vocab_size(self):
        if self.oov_dict is not None:
            return self.oov_dict.ext_vocab_size
        return None


def simple_tokenizer(text: str, lower: bool = False, newline: str = None) -> List[str]:
    """Split an already tokenized input `text`."""
    if lower:
        text = text.lower()
    if newline is not None:  # replace newline by a token
        text = text.replace("\n", " " + newline + " ")
    return text.split()


class Dataset(object):
    def __init__(
        self,
        filename: str,
        tokenize: Callable = simple_tokenizer,
        max_src_len: int = None,
        max_tgt_len: int = None,
        truncate_src: bool = False,
        truncate_tgt: bool = False,
    ):
        print("Reading dataset %s..." % filename, end=" ", flush=True)
        self.filename = filename
        self.pairs = []
        self.src_len = 0
        self.tgt_len = 0
        dataset_open = open
        if filename.endswith(".gz"):
            dataset_open = gzip.open
        with dataset_open(filename, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                pair = line.strip().split("\t")
                if len(pair) != 2:
                    print("Line %d of %s is malformed." % (i, filename))
                    continue
                src = tokenize(pair[0])
                if max_src_len and len(src) > max_src_len:
                    if truncate_src:
                        src = src[:max_src_len]
                    else:
                        continue
                tgt = tokenize(pair[1])
                if max_tgt_len and len(tgt) > max_tgt_len:
                    if truncate_tgt:
                        tgt = tgt[:max_tgt_len]
                    else:
                        continue
                src_len = len(src) + 1  # EOS
                tgt_len = len(tgt) + 1  # EOS
                self.src_len = max(self.src_len, src_len)
                self.tgt_len = max(self.tgt_len, tgt_len)
                self.pairs.append(Example(src, tgt, src_len, tgt_len))
        # print("%d pairs." % len(self.pairs))

    def build_vocab(
        self,
        vocab_size: int = None,
        src: bool = True,
        tgt: bool = True,
        embed_file: str = None,
    ) -> Vocab:
        filename, _ = os.path.splitext(self.filename)
        if vocab_size:
            filename += ".%d" % vocab_size
        filename += ".vocab"
        if os.path.isfile(filename):
            vocab = torch.load(filename)
            # print("Vocabulary loaded, %d words." % len(vocab))
        else:
            print("Building vocabulary...", end=" ", flush=True)
            vocab = Vocab()
            for example in self.pairs:
                if src:
                    vocab.add_words(example.src)
                if tgt:
                    vocab.add_words(example.tgt)
            vocab.trim(vocab_size=vocab_size)
            print("%d words." % len(vocab))
            torch.save(vocab, filename)
        if embed_file:
            count = vocab.load_embeddings(embed_file)
            # print("%d pre-trained embeddings loaded." % count)
        return vocab

    def generator(
        self,
        batch_size: int,
        src_vocab: Vocab = None,
        tgt_vocab: Vocab = None,
        ext_vocab: bool = False,
    ):
        ptr = len(self.pairs)  # make sure to shuffle at first run
        if ext_vocab:
            assert src_vocab is not None
            base_oov_idx = len(src_vocab)
        while True:
            if ptr + batch_size > len(self.pairs):
                shuffle(self.pairs)  # shuffle inplace to save memory
                ptr = 0
            examples = self.pairs[ptr : ptr + batch_size]
            ptr += batch_size
            src_tensor, tgt_tensor = None, None
            lengths, oov_dict = None, None
            if src_vocab or tgt_vocab:
                # initialize tensors
                if src_vocab:
                    examples.sort(key=lambda x: -x.src_len)
                    lengths = [x.src_len for x in examples]
                    max_src_len = lengths[0]
                    src_tensor = torch.zeros(max_src_len, batch_size, dtype=torch.long)
                    if ext_vocab:
                        oov_dict = OOVDict(base_oov_idx)
                if tgt_vocab:
                    max_tgt_len = max(x.tgt_len for x in examples)
                    tgt_tensor = torch.zeros(max_tgt_len, batch_size, dtype=torch.long)
                # fill up tensors by word indices
                for i, example in enumerate(examples):
                    if src_vocab:
                        for j, word in enumerate(example.src):
                            idx = src_vocab[word]
                            if ext_vocab and idx == src_vocab.UNK:
                                idx = oov_dict.add_word(i, word)
                            src_tensor[j, i] = idx
                        src_tensor[example.src_len - 1, i] = src_vocab.EOS
                    if tgt_vocab:
                        for j, word in enumerate(example.tgt):
                            idx = tgt_vocab[word]
                            if ext_vocab and idx == src_vocab.UNK:
                                idx = oov_dict.word2index.get((i, word), idx)
                            tgt_tensor[j, i] = idx
                        tgt_tensor[example.tgt_len - 1, i] = tgt_vocab.EOS
            yield Batch(examples, src_tensor, tgt_tensor, lengths, oov_dict)


class Hypothesis(object):
    def __init__(
        self, tokens, log_probs, dec_hidden, dec_states, enc_attn_weights, num_non_words
    ):
        self.tokens = tokens  # type: List[int]
        self.log_probs = log_probs  # type: List[float]
        self.dec_hidden = dec_hidden  # shape: (1, 1, hidden_size)
        self.dec_states = dec_states  # list of dec_hidden
        self.enc_attn_weights = enc_attn_weights  # list of shape: (1, 1, src_len)
        self.num_non_words = num_non_words  # type: int

    def __repr__(self):
        return repr(self.tokens)

    def __len__(self):
        return len(self.tokens) - self.num_non_words

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.log_probs)

    def create_next(
        self, token, log_prob, dec_hidden, add_dec_states, enc_attn, non_word
    ):
        return Hypothesis(
            tokens=self.tokens + [token],
            log_probs=self.log_probs + [log_prob],
            dec_hidden=dec_hidden,
            dec_states=self.dec_states + [dec_hidden]
            if add_dec_states
            else self.dec_states,
            enc_attn_weights=self.enc_attn_weights + [enc_attn]
            if enc_attn is not None
            else self.enc_attn_weights,
            num_non_words=self.num_non_words + 1 if non_word else self.num_non_words,
        )


def show_plot(
    loss, step=1, val_loss=None, val_metric=None, val_step=1, file_prefix=None
):
    plt.figure()
    fig, ax = plt.subplots(figsize=(12, 8))
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    ax.set_ylabel("Loss", color="b")
    ax.set_xlabel("Batch")
    plt.plot(range(step, len(loss) * step + 1, step), loss, "b")
    if val_loss:
        plt.plot(range(val_step, len(val_loss) * val_step + 1, val_step), val_loss, "g")
    if val_metric:
        ax2 = ax.twinx()
        ax2.plot(
            range(val_step, len(val_metric) * val_step + 1, val_step), val_metric, "r"
        )
        ax2.set_ylabel("ROUGE", color="r")
    if file_prefix:
        plt.savefig(file_prefix + ".png")
        plt.close()


def show_attention_map(src_words, pred_words, attention, pointer_ratio=None):
    fig, ax = plt.subplots(figsize=(16, 4))
    im = plt.pcolormesh(np.flipud(attention), cmap="GnBu")
    # set ticks and labels
    ax.set_xticks(np.arange(len(src_words)) + 0.5)
    ax.set_xticklabels(src_words, fontsize=14)
    ax.set_yticks(np.arange(len(pred_words)) + 0.5)
    ax.set_yticklabels(reversed(pred_words), fontsize=14)
    if pointer_ratio is not None:
        ax1 = ax.twinx()
        ax1.set_yticks(
            np.concatenate([np.arange(0.5, len(pred_words)), [len(pred_words)]])
        )
        ax1.set_yticklabels("%.3f" % v for v in np.flipud(pointer_ratio))
        ax1.set_ylabel("Copy probability", rotation=-90, va="bottom")
    # let the horizontal axes labelling appear on top
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    # rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=-45, ha="right", rotation_mode="anchor")


non_word_char_in_word = re.compile(r"(?<=\w)\W(?=\w)")
not_for_output = {"<PAD>", "<SOS>", "<EOS>", "<UNK>"}


def format_tokens(
    tokens: List[str], newline: str = "<P>", for_rouge: bool = False
) -> str:
    """Join output `tokens` for ROUGE evaluation."""
    tokens = filter(lambda t: t not in not_for_output, tokens)
    if for_rouge:
        tokens = [non_word_char_in_word.sub("", t) for t in tokens]  # "n't" => "nt"
    if newline is None:
        s = " ".join(tokens)
    else:  # replace newline tokens by newlines
        lines, line = [], []
        for tok in tokens:
            if tok == newline:
                if line:
                    lines.append(" ".join(line))
                line = []
            else:
                line.append(tok)
        if line:
            lines.append(" ".join(line))
        s = "\n".join(lines)
    return s


BAD_ALIAS = [
    "running quality",
    "avg count",
    "ccl quality",
    "pit pit",
    "8 ittt",
    "minute kicks",
    "of correction",
    "maxerrgrkt api",
    "amplitude pipe",
    "minimum in",
    "permeability minerrork",
    "maxerrgrkt gamma",
    "photoelectric maxerroru",
    "cement cement",
    "spectral cchl",
    "spectral cti",
    "spectral csi",
    "spectral cfe",
    "spectral ic",
    "spectral cgd",
    "spectral csul",
    "spectral chy",
    "spectral cca",
    "spectral erdf",
    "permeability essr",
    "spectral ck",
    "deep f",
    "spontaneous analog",
    "gamma a",
    "gamma b",
    "caliper medium",
    "medium cnts",
    "ewr resistivity",
    "tension head",
    "medium using",
    "spectral enra",
    '1" 19',
    '2" quality',
    "neutron **merged**",
    "5 gamma",
    "medium freq",
    "from echo",
    "calibrated calibrated",
    "echo quality",
    "medium corr.)",
    "spectral total_gas",
    "fluid fluid",
    "photoelectric fluid",
    "fluid quality",
    "medium porosity",
    "6 porosity",
    "spectral g/r",
    "density %",
    "spectral tbit10",
    "spontaneous spcg:2",
    "spontaneous spcg:1",
    "from correction",
    "from lithology",
    "16 lithology",
    "3 ray",
    "gamma v/v",
    "gamma grpo:1",
    "neutron vls",
    "sonic sprl:1",
    "spectral unknown:17",
    "spectral unknown:19",
    "spectral unknown:20",
    "spectral unknown:3",
    "spectral unknown:21",
    "spectral unknown:26",
    "spectral unknown:16",
    "spectral unknown:24",
    "spectral unknown:12",
    "spectral unknown:27",
    "spectral unknown:4",
    "spectral unknown:5",
    "spectral unknown:6",
    "spectral unknown:14",
    "spectral unknown:25",
    "spectral unknown:18",
    "spectral unknown:7",
    "spectral unknown:2",
    "spectral unknown:10",
    "spectral unknown:1",
    "spectral unknown:11",
    "spectral unknown:23",
    "spectral unknown:13",
    "spectral unknown:15",
    "spectral unknown:9",
    "spectral unknown:8",
    "spectral unknown:22",
    "dtl lithology",
    "density gre",
    "medium 3ft",
    "medium 5ft",
    "medium mult-lo",
    "medium fb8",
    "medium background",
    "medium sigma",
    "borehole filter",
    "medium pop",
    "caliper nb6",
    "neutron nprl:1",
    "density dprl:1",
    "amplitude ampsum",
    "neutron 10",
    "8 porosity",
    "caliper 22",
    "deep flg",
    "caliper dec",
    "medium c.",
    "dpss quality",
    "medium min/ft",
    "deep fluoresence",
    "caliper lithology",
    "gamma bcorr",
    "farquality quality",
    "permeability depth.",
    "a34h medium",
    "slowness quality",
    "spectral dphir",
    "minimum attenuation",
    "amplitude maxampl",
    "bc resistivity",
    "density b-quad",
    "lcrb porosity",
    "atten medium",
    "phase shallow",
    "deep projection",
    "gamma 6psi",
    "slowness resistivity",
    "density waveform",
    "5 medium",
    "neutron trace1",
    "density acou",
    "atten shallow",
    "medium avg",
    "phase deep",
    "permeability rhnt",
    "of saturation",
    "errors medium",
    "lbf porosity",
    "borehole porosity",
    "photoelectric uranium",
    "factor correction",
    "spectral bsd1",
    "deep {f13.4}",
    "medium 1in",
    "ae60 medium",
    "caliper scatter",
    "spectral lsd1",
    "filtered medium",
    "main medium",
    "mode quality",
    "temperature {f13.4}",
    "permeability rcnt",
    "temperature porosity",
    "permeability rhft",
    "permeability rcft",
    "ae20 resistivity",
    "spectral factor",
    "time resistivity",
    "shallow {f13.4}",
    "amplitude ampcal",
    "spectral bitsize",
    "spectral dfcal",
    "neutron a5dbhc",
    "spectral dphil",
    "neutron (imag)",
    "neutron (real)",
    "temperature 2",
    "raw quality",
    "deep inverse",
    "caliper mel",
    "spectral rwa2",
    "hngs spectral",
    "spectral r",
    "density deg",
    "amplitude amptemp",
    "evr gamma",
    "spectral abhv2",
    "shallow {f12.3}",
    "vol quality",
    "caliper t2",
    "amplitude amps8",
    "amplitude amps2",
    "amplitude amps7",
    "pefsa amplitude",
    "medium sector",
    "minimum sector",
    "amplitude amps6",
    "amplitude amps4",
    "amplitude amps1",
    "amplitude sector",
    "amplitude amps5",
    "deep q",
    "gamma aapi",
    "density {f11.4}",
    "spectral velocity",
    "spectral --millisecs.",
    "spectral c4/c5",
    "rt gamma",
    "mmho/m quality",
    "cond conductivity",
    "temperature {f11.4}",
    "caliper 10",
    "caliper {f11.4}",
    "3 density",
    "7 porosity",
    "amplitude amp3ft",
    "amplitude amp5ft",
    "spectral tg",
    "spectral bbinf",
    "spectral bviline",
    "spectral bpermf",
    "spectral clayline",
    "spectral bhflag",
    "neutron ngr",
    "gamma api-gr",
    "guard medium",
    "spectral rild2",
    "gapi {f13.4}",
    "2 kcl",
    "gamma 2",
    "- micronormal",
    "resistivity-am16 quality",
    "caliper counts",
    "spontaneous spcg",
    "correction correction",
    "density dlim",
    "decp density",
    "sonic pull",
    "density counts",
    "spectral sgin",
    "medium ohmm",
    "deep dil",
    "ft3 correction",
    "temperature density",
    "hi shallow",
    "hi resistivity",
    "lateral ohmm",
    "shallow one",
    "shallow weight",
    "caliper bondix",
    "amplitude 3'",
    "amplitude 5'",
    "microlog ohmm",
    "nsd1 nsd1",
    "nsdl nsdl",
    "permeability sgr2",
    "shear spectral",
    "lsn2 lsn2",
    "spherically resistivity",
    "rgc1 rgc1",
    "rgcn rgcn",
    "permeability gr01",
    "permeability gr06",
    "spectral tho2",
    "ls01 ls01",
    "medium sp02",
    "spectral tho1",
    "ll01 ll01",
    "hrd3 hrd3",
    "hrd4 hrd4",
    "spectral ssd1",
    "ssn1 ssn1",
    "spectral sft2/hrd2",
    "microlog pu",
    "medium dil",
    "near dga",
    "neutron cild",
    "spectral grf",
    "spectral tenf",
    "spectral calf",
    "permeability {f12.3}",
    "spectral tnra",
    "neutron {f13.4}",
    "depth base",
    "deep ohm-m",
    "caliper f3",
    "spontaneous spr",
    "spectral unknown",
    "medium cali",
    "caliper nvd",
    "lat quality",
    "spectral res:2",
    "lld ohmm",
    "density cn",
    "spectral rshal",
    "spectral rdeep",
    "sonic borehole",
    "gr ray",
    "gr api",
    "guard guard",
    "swn density",
    "near density",
    "neutron neu",
    "spectral sw6",
    "spectral sw7",
    "spectral sw5",
    "temperature 14",
    "5 quality",
    "cable quality",
    "medium small",
    "medium res(fl)",
    "foot quality",
    "density in",
    "a60 deep",
    "motor quality",
    "microlog mel15",
    "a60 resistivity",
    "line line",
    "dil medium",
    "permeability unknown",
    "spectralnoise spectralnoise",
    "permeability description",
    "gammatotal gamma",
    "deep hri",
    "density evr",
    "spontaneous 2",
    "based medium",
    "potential quality",
    "neutron 29",
    "density inches",
    "16 normal",
    "10 porosity",
    "neutron inches",
    "22 resistivity",
    "microloglateral ohmm",
    "medium fe",
    "spontaneous mv",
    "gamma grpo",
    "medium travel",
    "plus gamma",
    "tension tension",
    "shallow 14",
    "spectral 10",
    "9 resistivity",
    "permeability spectr.gamma-ray",
    "neutron neut",
    "5 ohmm",
    "sfl ohmm",
    "normal normal",
    "decp deep",
    "microlog tdt",
    "mll medium",
    "deep count",
    "water porosity",
    "medium epr",
    "6 ohmm",
    "14 quality",
    "sonic sonic",
    "hole hole",
    "ac1 us/f",
    "medium vil1",
    "medium vil2",
    "permeability gr2",
    "gr1 gr1",
    "neutron coun",
    "medium sp1",
    "ll1 ll1",
    "ls ls",
    "ls1 ls1",
    "6 deep",
    "densityporosity quality",
    "lsn1 lsn1",
    "cal3 in",
    "acou density",
    "caliper pe",
    "9 gamma",
    "medium res",
    "neutron (napi",
    "mv spontaneous",
    "filtered quality",
    "borehole quality",
    "shallow diameter",
    "vertical ohm-m",
    "neutron (napi)",
    "density {f13.4}",
    "borehole borehole",
    "waveform waveform",
    "caliper wall)",
    "spectral fluid",
    "atten resistivity",
    "ctn density",
    "photoelectric pe",
    "spectral poisson",
    "minimum minimum",
    "quality curve",
    "cal cal",
    "dtl quality",
    "of porosity",
    "permeability {f11.4}",
    "dtl processing",
    "deep mmo/m",
    "caliper #",
    "spectral minerrorth",
    "spectral maxerrorth",
    "minerrgrkt gamma",
    "maxerrgrtotal gamma",
    "minerrgrtotal api",
    "permeability maxerrork",
    "- inch",
    "permeability ft",
    "neutron (sandstone)",
    "medium api",
    "12 quality",
    "medium idsp",
    "spectral gsth",
    "spectral gsur",
    "6 resistivity",
    "factor resistivity",
    "18 gamma",
    "spontaneous mvolt",
    "deep #",
    "8 quality",
    "6 ray",
    "spectral pe",
    "rxo quality",
    "caliper deg",
    "medium waveform",
    "surface quality",
    "density ohmm",
    "shallow ohmm",
    "gamma gamma",
    "microlog in",
    "a30 quality",
    "gamma {f11.4}",
    "caliper res",
    "shear quality",
    "gamma {f13.4}",
    "and travel",
    "phase medium",
    "depth depth",
    "factor effect",
    "gradiomanometer porosity",
    "ppm gamma",
    "micro micro",
    "ppm spectral",
    "medium guard",
    "sonic volume",
    "mv mv",
    "minute mark",
    "medium processing",
    "tension quality",
    "deep 9",
    "a60 ray",
    "a30 medium",
    "fe quality",
    "rt shallow",
    "resolution resistivity",
    "neutron (napi,",
    "shallow ind.",
    "medium map",
    "factor quality",
    "tvd tvd",
    "formation quality",
    "6 effect",
    "cond quality",
    "correction quality",
    "far correction",
    "neutron spaced",
    "microlognormal ohmm",
    "g/cc correction",
    "rxrt quality",
    "microlog 16",
    "medium kev)",
    "ohm-m ohm-m",
    "pulse resistivity",
    "ll8 quality",
    "from quality",
    "medium medium",
    "deep ohmm",
    "gapi gapi",
    "of resistivity",
    "medium {f11.4}",
    "shallow curve",
    "from medium",
    "photoelectric photoelectric",
    "% quality",
    "far medium",
    "phase resistivity",
    "ppm quality",
    "deep processing",
    "caliper {f13.4}",
    "decp quality",
    "far quality",
    "spectral empty",
]
