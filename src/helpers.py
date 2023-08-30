import os
import logging

import torch
import numpy as np
from torch.utils.data import Dataset

# map a label to a string
#label2str = {1: "PER", 2: "LOC", 3: "ORG", 4: "MISC", 5: "O"}
label2str = {0: "<PAD>",
             1: "PER",
             2: "LOC",
             3: "ORG",
             4: "MISC",
             5: "O"}

# predefine a label_set: PER - 0, LOC - 1, ORG - 2, MISC - 3, O - 4
# labels_map = {'B-ORG': 3, 'O': 5, 'B-MISC': 4, 'B-PER': 1, 'I-PER': 1, 'B-LOC': 2, 'I-ORG': 3, 'I-MISC': 4, 'I-LOC': 2}
# labels_map = {'B-ORG': 2, 'O': 4, 'B-MISC': 3, 'B-PER': 0, 'I-PER': 0, 'B-LOC': 1, 'I-ORG': 2, 'I-MISC': 3, 'I-LOC': 1}
labels_map = {'<PAD>': 0, 'O': 5, 'B-PER': 1, 'I-PER': 1, 'B-ORG': 3, 'I-ORG': 3, 'B-LOC': 2, 'I-LOC': 2, 'B-MISC': 4, 'I-MISC': 4}

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
DEVICE = torch.device('cpu')

# TODO we can make this faster, or run it once and pickle the data
# TODO move from removal of all
def load_data2labels(input_file, directory, max_len):
    logging.info('loading data from: {}'.format(input_file))
    seq_set = []
    seq = []
    seq_set_label = []
    seq_label = []
    seq_set_len = []
    file_rel_path = directory + input_file
    script_path = os.path.dirname(__file__)
    file_full_path = os.path.join(script_path, file_rel_path)
    with open(file_full_path, "r", encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == "":
                # only store sequences with lenght <= max_len!
                if len(seq_label) <= max_len:
                    seq_set.append(" ".join(seq))
                    # Store labels as np arrays
                    seq_set_label.append(np.array(seq_label))
                    seq_set_len.append(len(seq_label))
                seq = []
                seq_label = []
            else:
                tok, label = line.split()
                seq.append(tok)
                seq_label.append(labels_map[label])
    return [seq_set, seq_set_label, seq_set_len]


def remove_language_prefix(embedding):
    return embedding[:]


# TODO pickle the output of this function to make the performance faster
# TODO optimize the runtime performance of this function
def load_crosslingual_embeddings(input_file, directory, vocab, max_vocab_size=20000, emb_size=40):
    script_path = os.path.dirname(__file__)
    dir_emb = '/' + directory.split('/')[0] + '/'
    file_full_path = script_path + dir_emb +  "emb/" + input_file
    embeddings = list(open(file_full_path, "r", encoding="utf-8").readlines())
    # Pre-process to remove the language prefix
    embeddings = map(remove_language_prefix, embeddings[1:])
    pre_w2v = {}
    vocabs_not_in_embedding = 0
    for emb in embeddings:
        parts = emb.strip().split()
        # Make sure embeddings have the correct dimensions
        assert emb_size == (len(parts) - 1)
        w = parts[0]
        vec = np.array(parts[1:], dtype=float)
        pre_w2v[w] = vec
    n_dict = len(vocab)
    if n_dict > max_vocab_size:
        logging.info('Vocabulary size is larger than {}'.format(max_vocab_size))
        raise SystemExit
    vocab_w2v = np.zeros((max_vocab_size, emb_size))
    for w, i in vocab.items():
        if w in pre_w2v:
            vocab_w2v[i] = pre_w2v[w]
        else:
            vocabs_not_in_embedding += 1
            vocab_w2v[i] = list(np.random.uniform(-0.25, 0.25, emb_size))
    logging.info('Vocabulary {} Embedding size {}'.format(n_dict, emb_size))
    logging.info('Vocabulary not in embedding file: {}'.format(vocabs_not_in_embedding))
    return vocab_w2v


def data2sents(X, Y):
    # data = []
    # for i in range(len(Y)):
    #     sent = []
    #     text = X[i]
    #     items = text.split()
    #     for j in range(len(Y[i])):
    #         sent.append((items[j], str(Y[i][j])))
    #     data.append(sent)

    data = []
    items = X.split()
    for j in range(len(Y)):
        data.append((items[j], str(Y[j])))
    return data


def get_mask_torch(batch_tensor):
    mask = batch_tensor.eq(0)
    mask = mask.eq(0)
    return mask

def get_mask_numpy(batch_array):
    mask = np.not_equal(batch_array, np.zeros_like(batch_array))
    return mask


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def lr_reset(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


class TrainDataset(Dataset):
    def __init__(self, story, end_index=None):
        self.end_idx = len(story[2])-1 if end_index is None else end_index
        self.sent = story[0][:self.end_idx+1]
        self.tag = story[1][:self.end_idx+1]
        self.sent_len = story[2][:self.end_idx+1]

    def __len__(self):
        return len(self.sent_len)

    def __getitem__(self, index):
        return self.sent[index], self.tag[index], self.sent_len[index]

    def append_story(self, story):
        # add any new queried data to the training dataset
        self.sent = np.append(self.sent, story[0], axis=0)
        self.tag = np.append(self.tag, story[1], axis=0)
        self.sent_len = np.append(self.sent_len, story[2])


class TestDataset(Dataset):
    def __init__(self, story):
        # TODO: check if sort is proper
        # TODO: check if mask is created properly
        sort_idx = (-story[2]).argsort()
        self.sent = story[0][sort_idx]
        self.tag = story[1][sort_idx]
        self.sent_len = story[2][sort_idx]
        self.mask = get_mask_numpy(self.sent)

    def __len__(self):
        return len(self.sent_len)

    def __getitem__(self, index):
        return self.sent[index], self.tag[index], self.sent_len[index], self.mask[index]