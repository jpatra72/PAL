import copy
import logging
import math
import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import helpers
from helpers import label2str, labels_map, DEVICE

# Compute f-score
# TODO use masks to remove the remaining loop
# TODO can run this vectorized function on the GPU for faster performance
def compute_fscore_vectorized(Y_pred, Y_true, sent_len):
    pre = 0
    pre_tot = 0
    rec = 0
    rec_tot = 0
    corr = 0
    total = 0
    number_of_examples = len(Y_true)
    for i in range(number_of_examples):
        # TODO path sentence length as a third parameter
        sentence_length = sent_len[i]
        true_array = Y_true[i][:sentence_length]
        pred_array = Y_pred[i][:sentence_length]
        is_correct = (pred_array == true_array)
        is_positive = (true_array != helpers.labels_map['O'])
        corr += np.sum(is_correct)
        rec_tot += np.sum(is_positive)
        rec += np.sum(is_positive & is_correct)
        is_predicted = pred_array != helpers.labels_map['O']
        pre_tot += np.sum(is_predicted)
        pre += np.sum(is_predicted & is_correct)
        total += sentence_length
    acc = corr * 1. / total
    logging.debug('Accuracy (token level) {}'.format(acc))
    if pre_tot == 0:
        pre = 0
    else:
        pre = 1. * pre / pre_tot
    rec = 1. * rec / rec_tot
    beta = 1
    f1score = 0
    if pre != 0 or rec != 0:
        f1score = (beta * beta + 1) * pre * rec / \
                  (beta * beta * pre + rec)
    logging.debug('Precision {0:.6f}; Recall {1:.6f}; F1 {2:.6f}'.format(pre, rec, f1score))
    return f1score, acc


class nerLSTM(nn.Module):
    def __init__(self,
                 dropout,
                 lstm_hidden_dim,
                 max_len=120,
                 embedding=None,
                 tag_count=6):

        super(nerLSTM, self).__init__()
        self.seq_len = max_len
        self.drop = nn.Dropout(dropout)
        self.input_dim = len(embedding[0])
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding, dtype=torch.float32))
        self.lstm = nn.LSTM(self.input_dim, lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(lstm_hidden_dim * 2, tag_count)

    def neg_log_likelihood_loss(self, input_sent, input_label, sen_lens):
        batch_size = input_sent.shape[0]
        seq_len = input_sent.shape[1]
        # get prob distr of sentence words over tags
        feature_out = self.probs_prediction(input_sent, sen_lens)
        feature_out = feature_out.contiguous().view(batch_size * seq_len, -1)
        # compute loss
        loss_function = nn.CrossEntropyLoss(ignore_index=labels_map['<PAD>'], reduction='sum')
        total_loss = loss_function(feature_out, input_label.contiguous().view(batch_size * seq_len))
        return total_loss

    def probs_prediction(self, input_sent, sen_lens):
        # get sent emb
        sent_emb = self.embedding(input_sent)
        sent_list = [sent_emb]
        sent_embedding = torch.cat(sent_list, 2)
        sent_drop = self.drop(sent_embedding)
        # pack sent for lstm processing
        packed_sens = pack_padded_sequence(sent_drop, sen_lens, True)
        hidden = None
        lstm_out, hidden = self.lstm(packed_sens, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out, total_length=self.seq_len)
        lstm_out = lstm_out.transpose(0, 1)
        feature_out = self.drop(lstm_out)
        # final fcnn to get prob distribution over tags
        feature_out = self.hidden2tag(feature_out)
        return feature_out

    def tagseq_prediction(self, input_sent, sen_lens, mask):
        batch_size = input_sent.shape[0]
        seq_len = input_sent.shape[1]
        feature_out = self.probs_prediction(input_sent, sen_lens)
        feature_out = feature_out.view(batch_size * seq_len, -1)
        tag_seq_probs, tag_seq = torch.max(feature_out, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        tag_seq = mask.long() * tag_seq
        return tag_seq_probs, tag_seq


class LSTMTagger(object):
    def __init__(self, dropout, lstm_hidden_dim, max_len, embedding_matrix, epochs, step_epoch_threshold, decay_rate, lr, momentum):
        logging.info("LSTM Tagger")
        self.name = 'LSTM'
        self.dropout = dropout
        self.lstm_hidden_dim = lstm_hidden_dim
        self.max_len = max_len
        self.embedding_matrix = embedding_matrix

        self.model = nerLSTM(dropout, lstm_hidden_dim, max_len, embedding_matrix, len(label2str))
        self.model_init_state_dict = copy.deepcopy(self.model.state_dict())
        self.model = self.model.to(DEVICE)

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.epochs = epochs
        self.step_epoch_threshold = step_epoch_threshold
        self.decay_r = decay_rate
        self.lr = lr
        self.momentum = momentum
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

    def train(self, dataloader, step):
        self.optimizer = helpers.lr_reset(self.optimizer, self.lr)
        self.model.train()
        for epoch in range(0, self.epochs):
            self.optimizer = helpers.lr_decay(self.optimizer, epoch, self.decay_r, self.lr)
            for x, y, slen in dataloader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                slen, sort_idx = slen.sort(descending=True)
                x, y = x[sort_idx], y[sort_idx]
                self.optimizer.zero_grad()
                # compute loss
                loss = self.model.neg_log_likelihood_loss(x, y, slen)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

            if step < self.step_epoch_threshold and epoch == self.epochs // 2:
                break

    def get_pred_conf(self, sent_idx, sent_len):
        if len(sent_idx.shape) == 3:
            sent_idx = torch.tensor(sent_idx).to(DEVICE)
        else:
            sent_idx = sent_idx[np.newaxis, :]
            sent_idx = torch.tensor(sent_idx).to(DEVICE)
        sent_len = torch.tensor(sent_len)
        # mask = helpers.get_mask(sent_idx)
        self.model.eval()
        with torch.no_grad():
            logit_preds = self.model.probs_prediction(sent_idx, sent_len)
            logit_preds = logit_preds[:, :, 1:].view(self.model.seq_len, -1)
            logprob_preds = self.log_softmax(logit_preds)
        prob_preds = torch.exp(logprob_preds)
        p_log_p = prob_preds[:sent_len[0]] * logprob_preds[:sent_len[0]]
        entropy = -p_log_p.sum(dim=1)
        avg_entropy = entropy.mean()
        return prob_preds.cpu().numpy(), avg_entropy.cpu().numpy()[np.newaxis]

    def test_score(self, dataloader):
        y_trues, y_preds, slens = [], [], []
        self.model.eval()
        for x, y_true, slen, mask in dataloader:
            x, mask = x.to(DEVICE), mask.to(DEVICE)
            with torch.no_grad():
                _, y_pred = self.model.tagseq_prediction(x, slen, mask)
            y_trues.extend(y_true.cpu().numpy())
            y_preds.extend(y_pred.cpu().numpy())
            slens.extend(slen.cpu().numpy())
        f1score, accuracy = compute_fscore_vectorized(y_preds, y_trues, slens)
        return f1score, accuracy

    def reboot(self):
        # self.model = nerLSTM(self.dropout, self.lstm_hidden_dim, self.max_len, self.embedding_matrix, len(label2str))
        self.model.load_state_dict(self.model_init_state_dict)
        self.model = self.model.to(DEVICE)
