import time
import logging

import wandb
import numpy as np
from attrdictionary import AttrDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from launcher_ner_bilingual import parse_args, initialize_game, construct_languages
from tagger import compute_fscore_vectorized, DEVICE, nerLSTM
from helpers import label2str, labels_map, get_mask_torch, lr_decay, lr_reset, TrainDataset, TestDataset


def data(game, budget):
    x_train = np.array(game.train_idx)
    y_train = np.array(game.train_idy)
    slen_train = np.array(game.train_slen)

    x_dev = game.dev_idx
    y_dev = game.dev_idy
    slen_dev = game.dev_slen

    x_test = game.test_idx
    y_test = game.test_idy
    slen_test = game.test_slen

    permutation = np.random.permutation(len(x_train))
    x_train = x_train[permutation]
    y_train = y_train[permutation]
    slen_train = slen_train[permutation]
    x_train = x_train[:budget]
    y_train = y_train[:budget]
    slen_train = slen_train[:budget]

    return (x_train, y_train, slen_train), (x_dev, y_dev, slen_dev), (x_test, y_test, slen_test)




def train_model(config, args):
    # load languages
    lang = construct_languages(args.train)
    lang = lang[0]

    # init game
    game = initialize_game(train_file=lang.train, test_file=lang.test, dev_file=lang.dev, emb_file=lang.emb,
                           data_dir=args.data_dir,
                           budget=args.budget, max_seq_len=args.max_seq_len, max_vocab_size=args.max_vocab_size,
                           emb_size=args.embedding_size, model_name=args.model_name)

    # init tagger model and optimizer
    model = nerLSTM(dropout=config.dropout,
                    lstm_hidden_dim=config.lhidden_dim,
                    embedding=game.w2v,
                    tag_count=len(label2str)).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)

    # load stories
    train_story, dev_story, test_story = data(game, args.budget)
    # process dev_story and load to device
    # x_test, y_test, slen_test = test_story
    # x_test = torch.tensor(x_test).to(DEVICE)
    # y_test = torch.tensor(y_test).to(DEVICE)
    # slen_test = torch.tensor(slen_test).to('cpu')
    # mask_test = get_mask_torch(x_test)
    # test_len = len(x_test)
    test_dataset = TestDataset(test_story)
    dataloader_test = DataLoader(test_dataset, batch_size=128, shuffle=False)

    train_time_aggregate = 0
    for step in range(0, args.budget):
        train_start = time.time()
        train_batch_size = config.batch_size
        #init dataloader
        train_dataset = TrainDataset(train_story, end_index=step)
        dataloader_train = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        # sort based on sen len
        model.train()
        # reset lr to initial value at the start of each step
        optimizer = lr_reset(optimizer, config.lr)
        for epoch in range(0, config.epochs):
            # decay lr inside epoch
            optimizer = lr_decay(optimizer, epoch, 0.05, config.lr)
            # minibatch training
            for x, y, slen in dataloader_train:
                x, y = x.to(DEVICE), y.to(DEVICE)
                slen, sort_idx = slen.sort(descending=True)
                x, y = x[sort_idx], y[sort_idx]
                optimizer.zero_grad()
                # compute loss
                loss = model.neg_log_likelihood_loss(x, y, slen)
                # prop loss
                loss.backward()
                clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            if step < config.step_epoch_threshold and epoch == config.epochs // 2:
                # when training data is low reduce epochs by half to avoid over-fitting, ie, when eps count < config.eps_epoch_threshold
                break
        train_time = time.time() - train_start
        train_time_aggregate += train_time

        # validation f1 score
        y_trues = []
        y_preds = []
        slens = []
        batch_size = 128
        model.eval()
        for x, y_true, slen, mask in dataloader_test:
            x, mask = x.to(DEVICE), mask.to(DEVICE)
            with torch.no_grad():
                _, y_pred = model.tagseq_prediction(x, slen, mask)
            y_trues.extend(y_true.cpu().numpy())
            y_preds.extend(y_pred.cpu().numpy())
            slens.extend(slen.cpu().numpy())

        f1score, accuracy = compute_fscore_vectorized(y_preds, y_trues, slens)
        # log score and losses for wandb hyperparam sweep
        wandb.log({'score': f1score,
                   'accuracy': accuracy,
                   'training_loss': loss,
                   'step_count': step,
                   'train_time': train_time_aggregate})

    return f1score, accuracy


def main():
    args = parse_args()
    # f1score, accuracy = train_model(config, args)
    wandb.init(project='lstm-tagger-sweep-test',
               config=args)

    f1score, accuracy = train_model(wandb.config, args)


if __name__=='__main__':
    # config = AttrDict({'dropout': 0.5,
    #                    'lhidden_dim': 40,
    #                    'lr': 0.15,
    #                    'momentum': 0.9,
    #                    'epochs':20,
    #                    'step_epoch_threshold': 32,
    #                    'batch_size':32})
    sweep_configuration = {
        'method': 'random',
        'metric':
            {
                'goal': 'maximize',
                'name': 'score'
            },
        'parameters':
            {
                'dropout': {'values': [0.1, 0.25, 0.5]},
                'lhidden_dim': {'values': [40, 60, 100]},
                'lr': {'values': [0.010, 0.015, 0.020]},
                'momentum': {'values': [0.9]},
                'epochs': {'values': [20, 40]},
                'step_epoch_threshold': {'values': [16, 32]},
                # 'dropout': {'values': [0.5]},
                # 'lhidden_dim': {'values': [40]},
                # 'lr': {'values': [0.015]},
                # 'momentum': {'values': [0.9]},
                # 'epochs': {'values': [20]},
                # 'step_epoch_threshold': {'values': [16]},
                'batch_size': {'values': [200]}
            }
    }

    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project='lstm-tagger-sweep-test'
    )
    wandb.agent(sweep_id, function=main, count=25)

    main()