"""
This file is for training the model of AlasKA from scratch,
which is a modification of ymfa/seq2seq-summarizer 
https://github.com/ymfa/seq2seq-summarizer/blob/master/
"""
import torch
import torch.nn as nn
import math
import os
from torch import optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from utils import Dataset, show_plot, Vocab, Batch
from model import Seq2Seq, DEVICE
from params import Params
from test import eval_batch


def train_batch(
    batch: Batch,
    model: Seq2Seq,
    criterion,
    optimizer,
    *,
    pack_seq=True,
    forcing_ratio=0.5,
    partial_forcing=True,
    sample=False,
    rl_ratio: float = 0,
    vocab=None,
    grad_norm: float = 0,
    show_cover_loss=False
):
    """
  :param batch: a train batch of a single example
  :param model: a pointer generator model
  :param criterion: negative log likelihood loss
  :param optimizer: Nadam or adam
  :param pack_seq: currently has no effect as batch size is 1
  :param forcing_ratio: initial percentage of using teacher forcing
  :param partial_forcing: in a seq, can some steps be teacher forced and some not?
  :param sample: are non-teacher forced inputs based on sampling or greedy selection?
  :param rl_ratio: use mixed objective if > 0; ratio of RL in the loss function
  :param vocab: vocabulary of the trained pointer generator
  :param grad_norm: use gradient clipping if > 0; max gradient norm
  :param show_cover_loss: include coverage loss in the loss shown in the progress bar?
  :return: loss of a batch

  Return loss of a batch
  """
    if not pack_seq:
        input_lengths = None
    else:
        input_lengths = batch.input_lengths

    optimizer.zero_grad()
    input_tensor = batch.input_tensor.to(DEVICE)
    target_tensor = batch.target_tensor.to(DEVICE)
    ext_vocab_size = batch.ext_vocab_size

    # create the seq2seq model
    out = model(
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

    loss = out.loss
    loss_value = out.loss_value

    loss.backward()
    if grad_norm > 0:
        clip_grad_norm_(model.parameters(), grad_norm)
    optimizer.step()

    target_length = target_tensor.size(0)
    return loss_value / target_length


def train(
    train_generator, vocab: Vocab, model: Seq2Seq, params: Params, valid_generator=None
):
    """
  :param train_generator: training dataset
  :param batch: a test batch of a single example
  :param vocab: vocabulary of the trained pointer generator
  :param model: a trained pointer generator
  :param params: parameters used for training
  :param valid_generator: validation dataset used during training
  :return: trained state dictionary and plot of loss

  Train a pointer generator.
  """
    # variables for plotting
    plot_points_per_epoch = max(math.log(params.n_batches, 1.6), 1.0)
    plot_every = round(params.n_batches / plot_points_per_epoch)
    plot_losses, cached_losses = [], []
    plot_val_losses, plot_val_metrics = [], []

    total_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print("Training %d trainable parameters..." % total_parameters)
    model.to(DEVICE)
    # initialize optimizer
    if params.optimizer == "adagrad":
        optimizer = optim.Adagrad(
            model.parameters(),
            lr=params.lr,
            initial_accumulator_value=params.adagrad_accumulator,
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=params.lr)
    past_epochs = 0
    total_batch_count = 0
    if params.lr_decay:
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, params.lr_decay_step, params.lr_decay, past_epochs - 1
        )
    criterion = nn.NLLLoss(ignore_index=vocab.PAD)
    best_avg_loss, best_epoch_id = float("inf"), None

    # initialize training
    for epoch_count in range(1 + past_epochs, params.n_epochs + 1):
        if params.lr_decay:
            lr_scheduler.step()
        rl_ratio = params.rl_ratio if epoch_count >= params.rl_start_epoch else 0
        epoch_loss, epoch_metric = 0, 0
        epoch_avg_loss, valid_avg_loss, valid_avg_metric = None, None, None
        prog_bar = tqdm(range(1, params.n_batches + 1), desc="Epoch %d" % epoch_count)
        model.train()

        # training batches
        for batch_count in prog_bar:
            # determine type of decay
            if params.forcing_decay_type:
                if params.forcing_decay_type == "linear":
                    forcing_ratio = max(
                        0,
                        params.forcing_ratio - params.forcing_decay * total_batch_count,
                    )
                elif params.forcing_decay_type == "exp":
                    forcing_ratio = params.forcing_ratio * (
                        params.forcing_decay ** total_batch_count
                    )
                elif params.forcing_decay_type == "sigmoid":
                    forcing_ratio = (
                        params.forcing_ratio
                        * params.forcing_decay
                        / (
                            params.forcing_decay
                            + math.exp(total_batch_count / params.forcing_decay)
                        )
                    )
                else:
                    raise ValueError(
                        "Unrecognized forcing_decay_type: " + params.forcing_decay_type
                    )
            else:
                forcing_ratio = params.forcing_ratio

            # train batch
            batch = next(train_generator)
            loss = train_batch(
                batch,
                model,
                criterion,
                optimizer,
                pack_seq=params.pack_seq,
                forcing_ratio=forcing_ratio,
                partial_forcing=params.partial_forcing,
                sample=params.sample,
                rl_ratio=rl_ratio,
                vocab=vocab,
                grad_norm=params.grad_norm,
                show_cover_loss=params.show_cover_loss,
            )
            # update loss
            epoch_loss += float(loss)
            epoch_avg_loss = epoch_loss / batch_count
            # update progress bar
            prog_bar.set_postfix(loss="%g" % epoch_avg_loss)

            cached_losses.append(loss)
            total_batch_count += 1
            if total_batch_count % plot_every == 0:
                period_avg_loss = sum(cached_losses) / len(cached_losses)
                plot_losses.append(period_avg_loss)
                cached_losses = []

        # validation batches
        if valid_generator is not None:
            valid_loss, valid_metric = 0, 0
            prog_bar = tqdm(
                range(1, params.n_val_batches + 1), desc="Valid %d" % epoch_count
            )
            model.eval()

            # update validation loss and progress bar
            for batch_count in prog_bar:
                batch = next(valid_generator)
                loss = eval_batch(
                    batch,
                    model,
                    vocab,
                    criterion,
                    pack_seq=params.pack_seq,
                    show_cover_loss=params.show_cover_loss,
                )
                valid_loss += loss
                valid_avg_loss = valid_loss / batch_count
                prog_bar.set_postfix(loss="%g" % valid_avg_loss)
            # plot validation loss
            plot_val_losses.append(valid_avg_loss)

        else:  # no validation, "best" is defined by training loss
            if epoch_avg_loss < best_avg_loss:
                best_epoch_id = epoch_count
                best_avg_loss = epoch_avg_loss

        if params.model_path_prefix:
            # save model
            filename = "%s.%02d.pt" % (params.model_path_prefix, epoch_count)
            torch.save(model.state_dict(), filename)
            if not params.keep_every_epoch:  # clear previously saved models
                for epoch_id in range(1 + past_epochs, epoch_count):
                    if epoch_id != best_epoch_id:
                        try:
                            prev_filename = "%s.%02d.state_dict.pt" % (
                                params.model_path_prefix,
                                epoch_id,
                            )
                            os.remove(prev_filename)
                        except FileNotFoundError:
                            pass
            # save training status
            torch.save(
                {
                    "epoch": epoch_count,
                    "total_batch_count": total_batch_count,
                    "train_avg_loss": epoch_avg_loss,
                    "valid_avg_loss": valid_avg_loss,
                    "valid_avg_metric": valid_avg_metric,
                    "best_epoch_so_far": best_epoch_id,
                    "params": params,
                    "optimizer": optimizer,
                },
                "%s.train.pt" % params.model_path_prefix,
            )

        # raise to power if greater than zero
        if rl_ratio > 0:
            params.rl_ratio **= params.rl_ratio_power

        # save plot to path
        show_plot(
            plot_losses,
            plot_every,
            plot_val_losses,
            plot_val_metrics,
            params.n_batches,
            params.model_path_prefix,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the AlasKA model.")
    p = Params()

    # build dataset
    dataset = Dataset(
        p.data_path,
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
    if p.val_data_path:
        val_dataset = Dataset(
            p.val_data_path,
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
    # val_gen = None

    train(train_gen, v, m, p, val_gen)
    # print sizes of layers in state dictionary
    print("Model's state_dict:")
    for param_tensor in m.state_dict():
        print(param_tensor, "\t", m.state_dict()[param_tensor].size())
    # save model to path
    torch.save(m.state_dict(), "state_dict.pth")
