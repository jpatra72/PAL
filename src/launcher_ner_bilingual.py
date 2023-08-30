import argparse
import logging
import os
import random as rn
import sys
import time
from datetime import datetime
from collections import defaultdict
from itertools import chain
from collections import namedtuple

import copy
import numpy as np

import wandb
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import helpers
import tagger
from game_ner import NERGame
from robot import RobotCNNDQN
from robot import RobotRandom
from tagger import LSTMTagger
from helpers import label2str
from helpers import TrainDataset

Language = namedtuple('Language', ['train', 'test', 'dev', 'emb', 'tagger'])


# TODO call by reference global variables!
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default="cnndqn", help="require a decision agent")
    parser.add_argument('--episode', type=int, default=10, help="maximum episode number for playing the game")
    parser.add_argument('--budget', type=int, default=10, help="budget for annotating")
    parser.add_argument('--train', type=str, default="en.train,en.testa,en.testb,cc.en.40.vec,models/en.model.saved",
                        help="training phase")
    parser.add_argument('--test', type=str, default="de.train,de.testa,de.testb,cc.de.40.vec,models/de.model.saved",
                        help="testing phase")
    parser.add_argument('--data_dir', type=str, default="data/conll2003/", help="path to the dataset folder")
    parser.add_argument('--wandb_tracking', type=bool, default=False, help='turn on wandb tracking')
    parser.add_argument('--wandb_project', type=str, default='PAL_LSTM', help='wandb project name')

    parser.add_argument('--max_seq_len', type=int, default=80, required=False, help='sequence')
    parser.add_argument('--max_vocab_size', type=int, default=20000, required=False, help='vocabulary')
    parser.add_argument('--embedding_size', type=int, default=40, required=False, help='embedding size')

    parser.add_argument('--model_name', type=str, default='LSTM', help='model name')
    parser.add_argument('--dropout', type=float, default=0.1, help='lstm dropout value')
    parser.add_argument('--lstm_hidden_dim', type=int, default=40, help='lstm hidden dimension')
    parser.add_argument('--epochs', type=int, default=20, help='train epochs for lstm every time a new label is queried')
    parser.add_argument('--step_epoch_threshold', type=int, default=32, help='step count until full value of epochs is used for training')
    parser.add_argument('--decay_rate', type=float, default=0.05, help='learning rate decay of the LSTM optimizer')
    parser.add_argument('--lr', type=float, default=0.02, help='lstm optimizer learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help="lstm optimizer's momentum factor")

    parser.add_argument('--replay_mem_size', type=int, default=1000, help='replay buffer size')
    parser.add_argument('--target_update_time', type=int, default=500, help='target network update time')
    parser.add_argument('--explore_steps', type=float, default=150000.0, help='epsilon scheduler factor')

    parser.add_argument('--log_path', type=str, required=False, default='logs/log.txt', help='log file path')
    # Log level
    parser.add_argument('--log_level', type=str, required=False, default='DEBUG', help='logging level')

    # model name
    return parser.parse_args()


def assert_list_of_sentences(sentences):
    assert type(sentences) == list
    assert len(sentences) >= 1
    assert type(sentences[0]) == str


def get_word_frequencies(sentences):
    words_dict = defaultdict(lambda: 0)
    for sentence in sentences:
        for word in sentence.split():
            words_dict[word] = words_dict[word] + 1
    return words_dict


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


# Returns ndarray mapping every word to an index
def sentences_to_idx(sentences, word_to_idx, max_len, pad_value, unk_value):
    unpadded_sequence = [torch.tensor([word_to_idx[w] if w in word_to_idx else unk_value for w in s.split()]) for s in
                         sentences]
    padded_sequence = pad_sequence(unpadded_sequence, batch_first=True, padding_value=pad_value)
    if padded_sequence.shape[1] < max_len:
        pad_cols = max_len - padded_sequence.shape[1]
        padded_sequence = F.pad(input=padded_sequence, pad=(0, pad_cols, 0, 0), mode='constant', value=pad_value)
    return np.array(padded_sequence.tolist())


def labels_to_idy(labels, max_len, num_tags, pad_value):
    labels = [torch.tensor(label) for label in labels]
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=pad_value)
    if padded_labels.shape[1] < max_len:
        pad_cols = max_len - padded_labels.shape[1]
        padded_labels = F.pad(input=padded_labels, pad=(0, pad_cols, 0, 0), mode='constant', value=pad_value)
    padded_labels_categorical = [to_categorical(i, num_classes=num_tags) for i in padded_labels]
    return np.array(padded_labels_categorical), np.array(padded_labels)


# Creates a vocabulary given iterator over sentences and maximum vocab size
def get_vocabulary(sentences, max_vocab_size):
    # assert max_vocab_size at least two
    assert max_vocab_size >= 2
    # Step 1 identify all unique words
    words = get_word_frequencies(sentences=sentences)
    logging.info('number of unique words before frequency trimming: {}'.format(len(words)))
    # Sort by frequency
    sorted_words = sorted(words.items(), key=lambda x: -x[1])
    # Only create vocab for the max_vocab_size entries, we don't need frequency anymore
    sorted_words = [w for (w, f) in sorted_words[:max_vocab_size - 2]]
    n_words = len(sorted_words)
    logging.info('number of unique words after frequency trimming: {}'.format(n_words))
    # TODO define these as constants
    # pad by zeros
    pad_value = 0
    # unk by n_words + 1
    unk_value = n_words + 1
    word_to_idx = {}
    word_to_idx['UNK'] = unk_value
    word_to_idx['PAD'] = pad_value
    for i, w in enumerate(sorted_words):
        word_to_idx[w] = i + 1
    return word_to_idx


def initialize_game(train_file, test_file, dev_file, emb_file, data_dir, budget, max_seq_len, max_vocab_size, emb_size,
                    model_name):
    # Load data
    logging.info('Loading data ..')
    train_x, train_y, train_lens = helpers.load_data2labels(input_file=train_file, directory=data_dir,
                                                            max_len=max_seq_len)
    test_x, test_y, test_lens = helpers.load_data2labels(input_file=test_file, directory=data_dir, max_len=max_seq_len)
    dev_x, dev_y, dev_lens = helpers.load_data2labels(input_file=dev_file, directory=data_dir, max_len=max_seq_len)
    logging.info('Processing data')
    # Build vocabulary
    logging.info('Max document length: {}'.format(max_seq_len))
    # Create vocabulary
    word_to_idx = get_vocabulary(sentences=chain(train_x, dev_x, test_x), max_vocab_size=max_vocab_size)
    pad_value = word_to_idx['PAD']
    unk_value = word_to_idx['UNK']
    # Train
    train_idx = sentences_to_idx(sentences=train_x, word_to_idx=word_to_idx, max_len=max_seq_len, pad_value=pad_value,
                                 unk_value=unk_value)
    num_tags = len(label2str)
    train_idy = labels_to_idy(labels=train_y, max_len=max_seq_len, num_tags=num_tags, pad_value=0)
    # Dev
    dev_idx = sentences_to_idx(sentences=dev_x, word_to_idx=word_to_idx, max_len=max_seq_len, pad_value=pad_value,
                               unk_value=unk_value)
    dev_idy = labels_to_idy(labels=dev_y, max_len=max_seq_len, num_tags=num_tags, pad_value=0)
    # Test
    test_idx = sentences_to_idx(sentences=test_x, word_to_idx=word_to_idx, max_len=max_seq_len, pad_value=pad_value,
                                unk_value=unk_value)
    test_idy = labels_to_idy(labels=test_y, max_len=max_seq_len, num_tags=num_tags, pad_value=0)
    # Build embeddings
    w2v = helpers.load_crosslingual_embeddings(input_file=emb_file, directory=data_dir, vocab=word_to_idx,
                                               max_vocab_size=max_vocab_size,
                                               emb_size=emb_size)
    # prepare story
    story = [train_x, train_y, train_idx, train_idy, np.array(train_lens)]
    logging.info('The length of the story {} (DEV = {}  TEST = {})'.format(len(train_x), len(dev_x), len(test_x)))
    test = [test_x, test_y, test_idx, test_idy, np.array(test_lens)]
    dev = [dev_x, dev_y, dev_idx, dev_idy, np.array(dev_lens)]
    # load game
    logging.info('Loading game ..')
    game = NERGame(story=story, dev=dev, test=test, max_len=max_seq_len, w2v=w2v, budget=budget, model_name=model_name)
    return game


def test_agent_batch(robot, game, model, budget, wandb_tracking):
    i = 0
    performance = []
    accuracy = []
    set_train_dataloader_flag = True

    game.reboot(model)
    while i < budget:
        sel_ind = game.current_frame
        # construct the observation
        observation = game.get_frame(model)
        action = robot.get_action(observation)
        if action == 1:
            sent_idx = game.train_idx[sel_ind]
            sent_idy = game.train_idy[sel_ind]
            sent_len = game.train_slen[sel_ind]
            story = ([sent_idx], [sent_idy], [sent_len])
            if set_train_dataloader_flag:
                dataset_train = TrainDataset(story)
                train_loader = DataLoader(dataset_train, batch_size=256, shuffle=True)
                set_train_dataloader_flag = False
            else:
                dataset_train.append_story(story)
            model.train(train_loader, i)
            f1score, acc = model.test_score(game.dataloader_test)
            performance.append(f1score)
            accuracy.append(acc)
            if wandb_tracking:
                wandb.log({"F1 Score Batch": f1score, "Accuracy Batch": acc, "custom step": i})
            i += 1
        game.current_frame += 1
    # train a crf and evaluate it
    model.train(train_loader, i)
    f1, acc = model.test_score(game.dataloader_test)
    performance.append(f1)
    accuracy.append(acc)
    logging.info('***TEST Batch F1 Score: {}'.format(performance))
    logging.info('***TEST Batch Accuracy: {}'.format(accuracy))


def test_agent_online(robot, game, model, budget, wandb_tracking):
    # how is this different from test_agent_batch? - the DQN network and the feature extractors of are both learned during active learning
    # and in test_agent_batch the feature extractors and dqn network is not learnt
    i = 0
    performance = []
    accuracy = []
    set_train_dataloader_flag = True
    game.reboot(model)
    while i < budget:
        sel_ind = game.current_frame
        observation = game.get_frame(model)
        action = robot.get_action(observation)
        if action == 1:
            sent_idx = game.train_idx[sel_ind]
            sent_idy = game.train_idy[sel_ind]
            sent_len = game.train_slen[sel_ind]
            story = ([sent_idx], [sent_idy], [sent_len])
            if set_train_dataloader_flag:
                dataset_train = TrainDataset(story)
                train_loader = DataLoader(dataset_train, batch_size=256, shuffle=True)
                set_train_dataloader_flag = False
            else:
                dataset_train.append_story(story)
            model.train(train_loader, i)
            f1score, acc = model.test_score(game.dataloader_test)
            performance.append(f1score)
            accuracy.append(acc)
            if wandb_tracking:
                wandb.log({"F1 Score Online": f1score, "Accuracy Online": acc, "custom step": i})
            i += 1
        reward, observation2, terminal = game.feedback(action, model)
        robot.update(observation, action, reward, observation2, terminal)
    model.train(train_loader, i)
    f1, acc = model.test_score(game.dataloader_test)
    performance.append(f1)
    accuracy.append(acc)
    logging.info('***TEST Online F1 Score: {}'.format(performance))
    logging.info('***TEST Online Accuracy: {}'.format(accuracy))


def build_model(model_name, max_len, embedding_matrix, dropout, lstm_hidden_dim,
                epochs, step_epoch_threshold, decay_rate, lr, momentum):
    if model_name == 'LSTM':
        model = LSTMTagger(dropout, lstm_hidden_dim, max_len, embedding_matrix,
                           epochs, step_epoch_threshold, decay_rate, lr, momentum)
    else:
        logging.error('Invalid model type')
        assert False
    return model


def play_ner(agent, train_lang, data_dir, budget, max_seq_len, max_vocab_size, embedding_size, max_episode, emb_size,
             model_name,
             dropout, lstm_hidden_dim, epochs, step_epoch_threshold, decay_rate, lr, momentum,
             replay_mem_size, target_update_time, explore_steps, wandb_tracking=False):
    train_lang_num = len(train_lang)
    actions = 2
    if agent == 'random':
        logging.info('Creating random robot...')
        robot = RobotRandom(actions)
    elif agent == 'dqn':
        # TODO Implement this
        assert False
    #        robot = RobotDQN(actions)
    elif agent == 'cnndqn':
        logging.info('Creating CNN DQN robot...')
        robot = RobotCNNDQN(actions, embedding_size=embedding_size, max_len=max_seq_len,
                            replay_memory_size=replay_mem_size, target_update_time=target_update_time, explore_steps=explore_steps)
    else:
        logging.info('** There is no robot.')
        raise SystemExit

    for i in range(train_lang_num):
        train = train_lang[i].train
        test = train_lang[i].test
        dev = train_lang[i].dev
        emb = train_lang[i].emb
        model_file = train_lang[i].tagger
        # initialize a NER game
        game = initialize_game(train, test, dev, emb, data_dir, budget, max_seq_len=max_seq_len,
                               max_vocab_size=max_vocab_size,
                               emb_size=emb_size, model_name=model_name)
        # initialize a decision robot
        robot.update_embeddings(game.w2v)
        # tagger
        model = build_model(model_name=model_name, max_len=max_seq_len,
                            embedding_matrix=game.w2v,
                            dropout=dropout, lstm_hidden_dim=lstm_hidden_dim,
                            epochs=epochs, step_epoch_threshold=step_epoch_threshold,
                            decay_rate=decay_rate, lr=lr, momentum=momentum)
        # play game
        episode = 1
        episode_steps = 1
        global_steps = 1
        logging.info('>>>>>> Playing game ..')
        gamma = 0.99
        episode_return = 0.0
        total_episodic_return = 0.0
        episode_start = time.time()
        total_episodic_time = 0
        test_score_old = 0
        while episode <= max_episode:
            # get observation dict from the raw data
            observation = game.get_frame(model)
            # query action from DQN agent based on observation
            action = robot.get_action(observation)
            logging.debug('> Action {}'.format(action))
            # run action and obtain reward and next observation
            reward, observation2, terminal = game.feedback(action, model)
            # compute return
            episode_return = gamma * episode_return + reward
            logging.debug('> Reward {}'.format(reward))
            # update sample to replay buffer
            robot.update(observation, action, reward, observation2, terminal)
            # increase episode steps
            episode_steps += 1
            global_steps += 1
            if terminal:
                total_episodic_return = total_episodic_return + episode_return
                average_episodic_return = total_episodic_return / float(episode)
                episode_time = time.time() - episode_start
                total_episodic_time = total_episodic_time + episode_time
                logging.info(
                    '>>>>>>> {0} / {1} episode return: {2:.4f}, average return: {3:.4f}, episode steps: {4}, episode time: {5:.2f}s, total time: {6:.2f}s, last f-score: {7:.4}'.format(
                        episode, max_episode, episode_return, average_episodic_return, episode_steps, episode_time,
                        total_episodic_time, float(game.performance)))
                if wandb_tracking:
                    wandb.log({"f1 score train": game.performance, "episode return": episode_return,
                               "average episodic return": average_episodic_return, "steps per episode": episode_steps,
                               "global steps": global_steps, "time per episode": episode_time, "total time": total_episodic_time})
                if episode > max_episode * 3 // 4 and game.performance > test_score_old and agent != 'random':
                    agent_state_dict = copy.deepcopy(robot.agent.state_dict())
                    test_score_old = game.performance
                # Reset return and time for next episode
                episode += 1
                global_steps += 1
                episode_steps = 0
                episode_return = 0.0
                episode_start = time.time()
    if agent != 'random':
        robot.agent.load_state_dict(agent_state_dict)
    return robot


def run_test(robot, test_lang, data_dir, budget, max_seq_len, max_vocab_size, emb_size, model_name,
             dropout, lstm_hidden_dim, epochs, step_epoch_threshold, decay_rate, lr, momentum, wandb_tracking=False):
    test_lang_num = len(test_lang)
    for i in range(test_lang_num):
        train = test_lang[i].train
        test = test_lang[i].test
        dev = test_lang[i].dev
        emb = test_lang[i].emb
        model_file = test_lang[i].tagger
        game2 = initialize_game(train, test, dev, emb, data_dir, budget, max_seq_len=max_seq_len,
                                max_vocab_size=max_vocab_size,
                                emb_size=emb_size, model_name=model_name)
        robot.update_embeddings(game2.w2v)
        model = build_model(model_name=model_name, max_len=max_seq_len,
                            embedding_matrix=game2.w2v,
                            dropout=dropout, lstm_hidden_dim=lstm_hidden_dim,
                            epochs=epochs, step_epoch_threshold=step_epoch_threshold,
                            decay_rate=decay_rate, lr=lr, momentum=momentum)

        test_agent_batch(robot, game2, model, budget, wandb_tracking)
        test_agent_online(robot, game2, model, budget, wandb_tracking)


def set_logger(log_path, log_level):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    dt = time.strftime("%Y-%m-%d_%H-%m-%S")
    log_path = log_path.split('.')
    log_path = f'{log_path[0]}_{dt}.{log_path[1]}'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)
    # Log to console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)


def construct_languages(all_langs, unique_id=None):
    # load the train data: source languages
    parts = all_langs.split(',')
    train_lang_num = int(len(parts) / 5)
    if len(parts) % 5 != 0:
        logging.info('Wrong inputs of training')
        raise SystemExit
    langs = []
    for i in range(train_lang_num):
        lang_i = i * 5
        train = parts[lang_i + 0]
        test = parts[lang_i + 1]
        dev = parts[lang_i + 2]
        emb = parts[lang_i + 3]
        tagger = parts[lang_i + 4]
        if unique_id is not None:
            tagger = tagger.split('/')
            tagger = f"{tagger[0]}/{unique_id}_{tagger[1]}"
        langs.append(Language(train=train, test=test, dev=dev, emb=emb, tagger=tagger))
    return langs


def fix_random_seeds():
    # fix random seed for numpy
    np.random.seed(153)
    # fix random seed for python random module
    rn.seed(165)
    # fix random seed for pytorch
    torch.manual_seed(1234)


def main():
    args = parse_args()

    run_name = f"{args.agent}_{args.budget}_{args.episode}_{int(time.time())}"
    wandb_prj = args.wandb_project
    if args.wandb_tracking:
        wandb.init(
            project=wandb_prj,
            config=vars(args),
            name=run_name,
        )

    set_logger(args.log_path, args.log_level)
    logging.info('working directory: {} \n'.format(os.getcwd()))
    logging.info('got args: ')
    logging.info(f"{args}\n")
    logging.info('fixing random seed, for full reproducibility, run on CPU and turn off multi-thread operations... \n')
    fix_random_seeds()
    dt = datetime.today().strftime('%m%d_%H%M%S')
    budget = args.budget
    train_lang = construct_languages(args.train, dt)
    # load the test data: target languages
    test_lang = construct_languages(args.test, dt)
    data_dir = args.data_dir
    max_seq_len = args.max_seq_len
    max_vocab_size = args.max_vocab_size
    embedding_size = args.embedding_size
    model_name = args.model_name
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logging.info('Device used for tensor ops: {}\n'.format(device))
    logging.info('Start of Robot Training')
    # play games for training a robot
    robot = play_ner(agent=args.agent, train_lang=train_lang, data_dir=data_dir, budget=budget, max_seq_len=max_seq_len,
                     max_vocab_size=max_vocab_size, embedding_size=embedding_size,
                     max_episode=args.episode, emb_size=embedding_size, model_name=model_name,
                     dropout=args.dropout, lstm_hidden_dim=args.lstm_hidden_dim, epochs=args.epochs,
                     step_epoch_threshold=args.step_epoch_threshold, decay_rate=args.decay_rate, lr=args.lr,
                     momentum=args.momentum, replay_mem_size=args.replay_mem_size, target_update_time=args.target_update_time,
                     explore_steps=args.explore_steps, wandb_tracking=args.wandb_tracking)
    # play a new game with the trained robot
    run_test(robot=robot, test_lang=test_lang, budget=budget, data_dir=data_dir, max_seq_len=max_seq_len,
             max_vocab_size=max_vocab_size,
             emb_size=embedding_size, model_name=model_name,
             dropout=args.dropout, lstm_hidden_dim=args.lstm_hidden_dim, epochs=args.epochs,
             step_epoch_threshold=args.step_epoch_threshold, decay_rate=args.decay_rate, lr=args.lr,
             momentum=args.momentum, wandb_tracking=args.wandb_tracking)
    logging.info('experiment completed! yay!')


if __name__ == '__main__':
    main()
