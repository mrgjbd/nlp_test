#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 18-10-16 上午11:07
# @Author  : guoxz
# @Site    : 
# @File    : main.py
# @Software: PyCharm
# @Description

import gluonnlp as nlp
import multiprocessing as mp
import itertools
import mxnet as mx
from mxnet import gluon, init
from classify.imdb import MyIMDB
from classify.rnn import BiRnn
from classify.train import train

ctx = mx.gpu()

train_dataset = MyIMDB('train')
test_dataset = MyIMDB('test')

tokenizer = nlp.data.SpacyTokenizer('en')
len_clip = nlp.data.ClipSequence(500)


def get_len(x):
    return len(x[0])


def clip_data(x):
    return (len_clip(tokenizer(x[0])), x[1])


def preprocess_dataset(dataset):
    pool = mp.Pool()
    dataset = pool.map(clip_data, dataset)
    lengths = pool.map(get_len, dataset)
    pool.close()
    return dataset, lengths


train_dataset, train_data_lengths = preprocess_dataset(train_dataset)
test_dataset, test_data_lengths = preprocess_dataset(test_dataset)

train_seq = [x[0] for x in train_dataset]
counter = nlp.data.count_tokens(list(itertools.chain.from_iterable(train_seq)))
vocab = nlp.Vocab(counter, max_size=10000, padding_token=None, bos_token=None, eos_token=None)

glove = nlp.embedding.GloVe(source='glove.6B.50d', embedding_root='/opt/disk1/nlp/word2vec')

vocab.set_embedding(glove)


def token_to_idx(x):
    return vocab[x[0]], x[1]


pool = mp.Pool()
train_dataset = pool.map(token_to_idx, train_dataset)
test_dataset = pool.map(token_to_idx, test_dataset)
pool.close()

batchify_fn = nlp.data.batchify.Tuple(nlp.data.batchify.Pad(),
                                      nlp.data.batchify.Stack())

batchify_sampler = nlp.data.FixedBucketSampler(lengths=train_data_lengths,
                                               batch_size=16,
                                               num_buckets=10,
                                               ratio=0.5,
                                               shuffle=True)

train_dataloader = gluon.data.DataLoader(train_dataset,
                                         batch_sampler=batchify_sampler,
                                         batchify_fn=batchify_fn)

test_dataloader = gluon.data.DataLoader(test_dataset,
                                        batch_sampler=batchify_sampler,
                                        batchify_fn=batchify_fn)

embeding_size = 50
vocab_size = len(vocab)
num_hiddens = 128
num_layers = 2
net = BiRnn(vocab_size=vocab_size,
            num_hiddens=num_hiddens,
            embed_size=embeding_size,
            num_layers=num_layers)
net.initialize(init.Xavier(), ctx)
net.embedding.weight.set_data(vocab.embedding.idx_to_vec)
net.embedding.collect_params().setattr('grad_req', 'null')
# layer = gluon.nn.Embedding(input_dim=vocab_size, output_dim=embeding_size)
# layer.initialize(init.Xavier(), ctx)
# layer.weight.set_data(vocab.embedding.idx_to_vec)

lr = 0.1
num_epochs = 10

trainer = gluon.Trainer(net.collect_params(), optimizer='sgd',
                        optimizer_params={'learning_rate': lr, 'wd': 0, 'momentum': 0})
loss = gluon.loss.SoftmaxCrossEntropyLoss()

train(train_dataloader, test_dataloader, loss, net, trainer, num_epochs, ctx)
