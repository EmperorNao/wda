from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
from os.path import join

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from dataset import Dataset

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from constants import LOCAL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def addData():
    print("Reading lines...")

    data = Dataset()
    data.from_csv(join(LOCAL, "data/dataset.csv"))
    return data



MAX_LENGTH = 40

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


def prepareData():
    data = addData()

    lang = Lang("rus")
    print("Got %s sentences" % len(data))

    data = data[(data['title_len'] < MAX_LENGTH) & (data['IsTrain'] != 0)]
    sents = list(data['title'])

    print("Trimmed to %s sentences" % len(data))
    print("Counting words...")
    for sent in sents:
        lang.addSentence(sent)
    print("Counted words:")
    print(lang.name, lang.n_words)
    return lang, data


def get_dataloader(batch_size):
    lang, dataset = prepareData()

    n = len(dataset)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    targets = np.array(dataset['target'], dtype=np.int32)

    for idx, sent in enumerate(dataset['title']):
        inp_ids = indexesFromSentence(lang, sent)
        inp_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids

    print(input_ids.shape)
    print(targets.shape)

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.FloatTensor(targets).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return lang, train_dataloader


class Model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

        self.out = nn.Linear(hidden_size, 1)

    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded)
        return self.out(hidden.permute(1, 0, 2)).reshape(-1)



def train_epoch(dataloader, encoder, encoder_optimizer, criterion):

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()

        predictions = encoder(input_tensor)

        loss = criterion(
            predictions,
            target_tensor
        )
        loss.backward()

        encoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def train(train_dataloader, encoder, n_epochs, learning_rate=0.001,
               print_every=100, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, encoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()


hidden_size = 128
batch_size = 32

lang, train_dataloader = get_dataloader(batch_size)

encoder = Model(lang.n_words, hidden_size).to(device)

train(train_dataloader, encoder, 80, print_every=1, plot_every=1)

