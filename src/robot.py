import copy
import logging
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gymnasium as gym
    from gymnasium.spaces import Box, Discrete, Dict
except ImportError:
    try:
        import gym
        from gym.spaces import Box, Discrete, Dict
    except ImportError:
        raise Exception("Unable to import gym or gymnasium")
from stable_baselines3.common.buffers import DictReplayBuffer

from helpers import DEVICE

# Hyper Parameters:
LR_RATE = 1e-3
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 32.  # timesteps to observe before training
# REPLAY_MEMORY_SIZE = 1000  # number of previous transitions to remember
BATCH_SIZE = 32  # size of minibatch
# FINAL_EPSILON = 0
# INITIAL_EPSILON = 0
# or alternative:
FINAL_EPSILON = 0.001  # final value of epsilon
INITIAL_EPSILON = 0.4  # starting value of epsilon
# UPDATE_TIME = 500
# EXPLORE = 150000.


# Robot interface
class Robot(object):
    def __init__(self):
        pass

    def update_embeddings(self, embeddings):
        pass

    def get_action(self, observation):
        pass

    def update(self, observation, action, reward, observation2, terminal):
        pass


# Random Robot
class RobotRandom(Robot):
    def __init__(self, actions):
        super().__init__()
        self.actions = actions

    def get_action(self, observation):
        return random.randrange(self.actions)


class CNNSentence(nn.Module):
    def __init__(self, actions, vocab_size, seq_len, emb_size, class_cnt, filter_sizes_sent, filter_sizes_pred,
                 filter_cnt_sent, filter_cnt_pred):
        super().__init__()

        self.actions = actions
        self.seq_len = seq_len
        self.filter_sizes_sent = filter_sizes_sent
        self.filter_sizes_pred = filter_sizes_pred

        self.filter_cnt_sent = filter_cnt_sent
        self.filter_cnt_pred = filter_cnt_pred

        dropout_keep_prob = 0.5
        rand_emb_weights = torch.randn(vocab_size, emb_size)
        self.w = nn.Embedding.from_pretrained(rand_emb_weights)
        # sentence processing
        self.conv_layers_sent = nn.ModuleList([
            nn.Conv2d(1, filter_cnt_sent, (fsize, emb_size), padding='valid') for fsize in filter_sizes_sent
        ])

        self.drop_sent = nn.Dropout(1 - dropout_keep_prob)
        self.drop_pred = nn.Dropout(1 - dropout_keep_prob)

        # prediction processing
        self.conv_layers_pred = nn.ModuleList([
            nn.Conv2d(1, filter_cnt_pred, (fsize, class_cnt), padding='valid') for fsize in filter_sizes_pred
        ])
        # merging computed features
        self.fc1_s = nn.Linear(384, 256)
        self.fc1_p = nn.Linear(20, 256)
        self.fc1_c = nn.Linear(1, 256)
        self.fc2 = nn.Linear(256, self.actions)

    def p_sentence(self, input_sent):
        pooled_outputs = []

        embedded_chars = self.w(input_sent)

        for i, filter_size in enumerate(self.filter_sizes_sent):
            # Convolution Layer
            x = self.conv_layers_sent[i](embedded_chars)
            x = F.relu(x)
            # Maxpooling over the outputs
            pooled = F.max_pool2d(x, (self.seq_len - filter_size + 1, 1))
            pooled_outputs.append(pooled)
        num_filters_total = self.filter_cnt_sent * len(list(self.filter_sizes_sent))
        out_pool = torch.cat(pooled_outputs, 3)
        out_pool_reshaped = torch.reshape(out_pool, [-1, num_filters_total])

        out_pool_reshaped = self.drop_sent(out_pool_reshaped)
        return out_pool_reshaped

    def p_prediction(self, input_pred):
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes_pred):
            x = self.conv_layers_pred[i](input_pred)
            # x = F.relu(x)
            pooled = F.avg_pool2d(x, (self.seq_len - filter_size + 1, 1))
            pooled_outputs.append(pooled)
        num_filters_total = self.filter_cnt_pred * len(list(self.filter_sizes_pred))
        out_pool = torch.cat(pooled_outputs, 3)
        out_pool_reshaped = torch.reshape(out_pool, [-1, num_filters_total])

        out_pool_reshaped = self.drop_pred(out_pool_reshaped)
        return out_pool_reshaped

    def qnetwork(self, input_sent, input_pred, input_confidence):
        feature_sent = self.p_sentence(input_sent)
        feature_pred = self.p_prediction(input_pred)
        feature_conf = input_confidence

        qvals = self.fc1_s(feature_sent) \
                + self.fc1_p(feature_pred) \
                + self.fc1_c(feature_conf)
        qvals = self.fc2(qvals)

        return qvals


def create_buffer(buffer_size, sent_len):
    obs_space = Dict({"sent": Box(float('-inf'), float('inf'), shape=(sent_len,)),
                      "conf": Box(float('-inf'), float('inf'), shape=(1,)),
                      "pred": Box(float('-inf'), float('inf'), shape=(sent_len, 5))
                      })
    action_space = Discrete(2)
    rp_buffer = DictReplayBuffer(buffer_size,
                                 obs_space,
                                 action_space,
                                 DEVICE,
                                 handle_timeout_termination=False)
    return rp_buffer


# CNNDQN Robot
class RobotCNNDQN(Robot):

    def __init__(self, actions=2, vocab_size=20000, max_len=120, embeddings=None, embedding_size=40,
                 replay_memory_size=1000, target_update_time=200, explore_steps=120000):
        super().__init__()
        logging.debug('Creating a robot: CNN-DQN')
        # replay memory
        # self.replay_memory = deque()
        self.time_step = 0
        self.action = actions
        self.w_embeddings = embeddings
        self.max_len = max_len
        self.num_classes = 5
        self.epsilon = INITIAL_EPSILON

        self.replay_memory_size = replay_memory_size
        self.target_update_time = target_update_time
        self.explore_steps = explore_steps

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_size = embedding_size

        self.filter_size_sent = [3, 4, 5]
        self.filter_size_pred = [3]
        self.channel_out_sent = 128
        self.channel_out_pred = 20

        self.replay_memory = create_buffer(self.replay_memory_size, self.max_len)

        self.agent = CNNSentence(self.action, self.vocab_size,
                                 self.max_len, self.embedding_size, self.num_classes,
                                 self.filter_size_sent, self.filter_size_pred,
                                 self.channel_out_sent, self.channel_out_pred).to(DEVICE)
        self.target_agent = copy.deepcopy(self.agent)

        self.optim = torch.optim.Adam(self.agent.parameters(), lr=LR_RATE)

    # TODO: not called anywhere but still needs updating
    def initialise(self, max_len, embeddings):
        self.max_len = max_len
        self.w_embeddings = embeddings
        self.vocab_size = len(self.w_embeddings)
        self.embedding_size = len(self.w_embeddings[0])

    def get_action(self, observation):
        logging.debug('DQN is smart.')
        self.current_state = observation

        if random.random() <= self.epsilon:
            action = random.randrange(self.action)
        else:
            sent = torch.tensor(self.current_state['sent'], device=DEVICE, dtype=torch.int).view(1, 1, -1)
            pred = torch.tensor(self.current_state['pred'], device=DEVICE, dtype=torch.float32).view(1, 1, -1,
                                                                                                     self.num_classes)
            conf = torch.tensor(self.current_state['conf'], device=DEVICE, dtype=torch.float32)
            self.agent.eval()
            with torch.no_grad():
                qvalue = self.agent.qnetwork(sent, pred, conf)
            action = np.argmax(qvalue.cpu().numpy())
        # epsilon scheduling
        if self.epsilon > FINAL_EPSILON and self.time_step > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / self.explore_steps
        return action

    def update(self, observation, action, reward, next_observation, terminal):
        # self.current_state = observation
        # new_state = observation2
        action = np.array(action)
        self.replay_memory.add(observation, next_observation, action, reward, terminal, 'na')

        # if self.time_step > OBSERVE and self.time_step > BATCH_SIZE:
        if self.time_step > OBSERVE:
            self.train_qnetwork()
        self.time_step += 1 # global step count

    def train_qnetwork(self):
        self.agent.train()
        minibatch = self.replay_memory.sample(BATCH_SIZE)
        with torch.no_grad():
            target_next_qvals = self.target_agent.qnetwork(
                minibatch.next_observations['sent'].type(torch.int).view(BATCH_SIZE, 1, -1),
                minibatch.next_observations['pred'].view(BATCH_SIZE, 1, -1, self.num_classes),
                minibatch.next_observations['conf'], )
        target_qvals = minibatch.rewards + GAMMA * (
                (1 - minibatch.dones) * target_next_qvals.max(1, keepdim=True).values)

        old_qvals = self.agent.qnetwork(minibatch.observations['sent'].type(torch.int).view(BATCH_SIZE, 1, -1),
                                        minibatch.observations['pred'].view(BATCH_SIZE, 1, -1, self.num_classes),
                                        minibatch.observations['conf'], )

        loss = F.mse_loss(target_qvals, old_qvals.gather(1, minibatch.actions))

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.agent.eval()

        if self.time_step % self.target_update_time == 0:
            self.target_agent.load_state_dict(self.agent.state_dict())

    def update_embeddings(self, embeddings):
        self.w_embeddings = embeddings
        self.vocab_size = len(self.w_embeddings)
        self.embedding_size = len(self.w_embeddings[0])
        logging.debug('Assigning new word embeddings')
        logging.debug('New size {}'.format(self.vocab_size))
        with torch.no_grad():
            self.agent.w.weight = nn.parameter.Parameter(
                torch.tensor(self.w_embeddings, dtype=torch.float32, device=DEVICE), False)
        self.time_step = 0
        self.replay_memory = create_buffer(self.replay_memory_size, self.max_len)
