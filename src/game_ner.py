import os
import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

import helpers
from helpers import TrainDataset, TestDataset, DEVICE
import tagger as tg


class NERGame:

    def __init__(self, story, test, dev, max_len, w2v, budget, model_name):
        # build environment
        # load data as story
        logging.info('Initializing the game:')
        # import story
        self.train_x, self.train_y, self.train_idx, (self.train_cy, self.train_idy), self.train_slen = story
        self.test_x, self.test_y, self.test_idx, (self.test_cdy, self.test_idy), self.test_slen= test
        self.dev_x, self.dev_y, self.dev_idx, (self.dev_cdy, self.dev_idy), self.dev_slen = dev
        # create test dataloader
        dataset_test = TestDataset((self.test_idx, self.test_idy, self.test_slen))
        self.dataloader_test = DataLoader(dataset_test, batch_size=1024, shuffle=False)
        self.max_len = max_len
        self.w2v = w2v
        logging.info('Story: length = {}'.format(len(self.train_x)))
        # self.order = list(range(0, len(self.train_x)))
        self.order = np.random.permutation(len(self.train_x))
        # if re-order, use random.shuffle(self.order)
        # load word embeddings, pretrained - w2v
        logging.info('Dictionary size {} Embedding size {}'.format(len(self.w2v), len(self.w2v[0])))
        # when queried times is 'budget', then stop
        self.budget = budget
        self.queried_times = 0
        # TODO use ndarrays for everything
        # let's start
        self.episode = 0
        # story frame
        self.current_frame = 0
        self.terminal = False
        self.make_query = False
        self.performance = 0.0
        self.create_dataloader_flag = True

    def get_frame(self, model):
        self.make_query = False
        sent_idx = self.train_idx[self.order[self.current_frame]]
        sent_len = self.train_slen[self.order[self.current_frame]]
        sent_len = [sent_len]
        predictions, entropy = model.get_pred_conf(sent_idx, sent_len)
        observation = {"sent": sent_idx, "conf": entropy, "pred": predictions}
        return observation

    # tagger = model
    def feedback(self, action, model):
        is_terminal = False
        if action == 1:
            self.make_query = True
            self.query()
            new_performance = self.get_performance(model)
            reward = new_performance - self.performance
            if new_performance != self.performance:
                self.performance = new_performance
        else:
            reward = 0.
        # next frame
        if self.queried_times == self.budget:
            self.terminal = True
            is_terminal = True
            # update special reward
            # reward = new_performance * 100
            # prepare the next game
            self.reboot(model)
            next_sentence_idx = self.train_idx[self.order[self.current_frame]]
            next_sentence_len = self.train_slen[self.order[self.current_frame]]
        else:
            self.terminal = False
            # next_sentence = self.train_x[self.order[self.current_frame + 1]]
            next_sentence_idx = self.train_idx[self.order[self.current_frame + 1]]
            next_sentence_len = self.train_slen[self.order[self.current_frame + 1]]
            self.current_frame += 1
        next_sentence_len = [next_sentence_len]
        predictions, entropy = model.get_pred_conf(next_sentence_idx, next_sentence_len)
        #shapes: sent - (120,), conf - (1,), pred - (120, 5)
        next_observation = {"sent": next_sentence_idx, "conf": entropy, "pred": predictions}
        return reward, next_observation, is_terminal

    def query(self):
        # incorporate dataloader here
        idx = self.order[self.current_frame]
        story = ([self.train_idx[idx]], [self.train_idy[idx]], [self.train_slen[idx]])
        if self.create_dataloader_flag is True:
            self.create_dataloader_flag = False
            self.dataset_train = TrainDataset(story)
            self.dataloader_train = DataLoader(self.dataset_train, batch_size=256, shuffle=True)
        else:
            self.dataset_train.append_story(story)
        self.queried_times += 1
        # logging.debug "Select:", sentence, labels
        logging.debug('> Queried times {}'.format(len(self.dataset_train)))

    def get_performance(self, tagger):
        tagger.train(self.dataloader_train, self.queried_times)
        f1score, accuracy = tagger.test_score(self.dataloader_test)
        return f1score

    def reboot(self, model):
        # restart story
        self.order = np.random.permutation(len(self.train_x))
        self.queried_times = 0
        self.terminal = False
        self.current_frame = 0
        self.episode += 1
        self.create_dataloader_flag = True
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        model.reboot()
        logging.debug('> Next episode {}'.format(self.episode))
