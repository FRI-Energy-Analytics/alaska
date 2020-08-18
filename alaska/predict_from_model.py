import tarfile
from typing import Tuple, List
import torch
import math
from utils import Vocab, OOVDict, Batch, format_tokens, Dataset
from model import DEVICE, Seq2SeqOutput, Seq2Seq
from params import Params


def decode_batch_output(
    decoded_tokens, vocab: Vocab, oov_dict: OOVDict
) -> List[List[str]]:
    """Convert word indices to strings."""
    decoded_batch = []
    if not isinstance(decoded_tokens, list):
        decoded_tokens = decoded_tokens.transpose(0, 1).tolist()
    for i, doc in enumerate(decoded_tokens):
        decoded_doc = []
        for word_idx in doc:
            if word_idx >= len(vocab):
                word = oov_dict.index2word.get((i, word_idx), "<UNK>")
            else:
                word = vocab[word_idx]
            decoded_doc.append(word)
            if word_idx == vocab.EOS:
                break
        decoded_batch.append(decoded_doc)
    return decoded_batch


def decode_batch(
    batch: Batch,
    model: Seq2Seq,
    vocab: Vocab,
    criterion=None,
    *,
    pack_seq=True,
    show_cover_loss=False
) -> Tuple[List[List[str]], Seq2SeqOutput]:
    """Test the `model` on the `batch`, return the decoded textual tokens and the Seq2SeqOutput."""
    if not pack_seq:
        input_lengths = None
    else:
        input_lengths = batch.input_lengths
    with torch.no_grad():
        input_tensor = batch.input_tensor.to(DEVICE)
        if batch.target_tensor is None or criterion is None:
            target_tensor = None
        else:
            target_tensor = batch.target_tensor.to(DEVICE)
        out = model(
            input_tensor,
            target_tensor,
            input_lengths,
            criterion,
            ext_vocab_size=batch.ext_vocab_size,
            include_cover_loss=show_cover_loss,
        )
        decoded_batch = decode_batch_output(out.decoded_tokens, vocab, batch.oov_dict)
    target_length = batch.target_tensor.size(0)
    out.loss_value /= target_length
    return decoded_batch, out


def decode_one(*args, **kwargs):
    """
  Same as `decode_batch()` but because batch size is 1, the batch dim in visualization data is
  eliminated.
  """
    decoded_batch, out = decode_batch(*args, **kwargs)
    decoded_doc = decoded_batch[0]
    if out.enc_attn_weights is not None:
        out.enc_attn_weights = out.enc_attn_weights[: len(decoded_doc), 0, :]
    if out.ptr_probs is not None:
        out.ptr_probs = out.ptr_probs[: len(decoded_doc), 0]
    return decoded_doc, out


def eval_bs_batch(
    batch: Batch,
    model: Seq2Seq,
    vocab: Vocab,
    *,
    pack_seq=True,
    beam_size=4,
    min_out_len=1,
    max_out_len=None,
    len_in_words=True,
    best_only=True,
    details: bool = True
):
    """
  :param batch: a test batch of a single example
  :param model: a trained summarizer
  :param vocab: vocabulary of the trained summarizer
  :param pack_seq: currently has no effect as batch size is 1
  :param beam_size: the beam size
  :param min_out_len: required minimum output length
  :param max_out_len: required maximum output length (if None, use the model's own value)
  :param len_in_words: if True, count output length in words instead of tokens (i.e. do not count
                       punctuations)
  :param best_only: if True, run ROUGE only on the best hypothesis instead of all `beam size` many
  :param details: if True, also return a string containing the result of this document
  :return: mnemonics and predicted label

  Use a trained summarizer to predict
  """
    assert len(batch.examples) == 1
    with torch.no_grad():
        input_tensor = batch.input_tensor.to(DEVICE)
        hypotheses = model.beam_search(
            input_tensor,
            batch.input_lengths if pack_seq else None,
            batch.ext_vocab_size,
            beam_size,
            min_out_len=min_out_len,
            max_out_len=max_out_len,
            len_in_words=len_in_words,
        )
    if best_only:
        to_decode = [hypotheses[0].tokens]
        probability = math.log(-hypotheses[0].avg_log_prob, 10)
    else:
        to_decode = [h.tokens for h in hypotheses]
    decoded_batch = decode_batch_output(to_decode, vocab, batch.oov_dict)
    if details:
        # predicted = format_tokens(decoded_batch[0])
        predict_lst = format_tokens(decoded_batch[0]).split()
        predicted = str(predict_lst[0]) + " " + str(predict_lst[1])
    else:
        predicted = None
    if details:
        mnem = format_tokens(batch.examples[0].src).split()[-1]
    return predicted, mnem, probability


def eval_bs(test_set: Dataset, vocab: Vocab, model: Seq2Seq, params: Params):
    """
    :param test_set: dataset of summaries
    :param vocab: vocabularies of model
    :param model: model to use
    :param params: parameter file to read from
    :return: dictionary of predicted outputs
    Predict labels from summaries
    """
    test_gen = test_set.generator(1, vocab, None, True if params.pointer else False)
    n_samples = int(params.test_sample_ratio * len(test_set.pairs))

    if params.test_save_results and params.model_path_prefix:
        result_file = tarfile.open("data/results.tgz", "w:gz")
    output, prob_output = {}, {}
    model.eval()
    for _ in range(1, n_samples + 1):
        batch = next(test_gen)
        predicted, mnem, prob = eval_bs_batch(
            batch,
            model,
            vocab,
            pack_seq=params.pack_seq,
            beam_size=params.beam_size,
            min_out_len=params.min_out_len,
            max_out_len=params.max_out_len,
            len_in_words=params.out_len_in_words,
            details=result_file is not None,
        )
        if predicted:
            output[mnem] = predicted
            prob_output[mnem] = prob
    return output, prob_output


def make_prediction(test_path):
    """
    :param test_path: path to LAS file
    :return: dictionary of mnemonic and label
    Make predictions using pointer generator
    """
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
    m.encoder.gru.flatten_parameters()
    m.decoder.gru.flatten_parameters()

    d = Dataset(test_path)
    output, prob_output = eval_bs(d, v, m, p)
    return output, prob_output
