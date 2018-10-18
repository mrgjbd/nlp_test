#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 18-10-15 下午6:11
# @Author  : guoxz
# @Site    : 
# @File    : rnn.py
# @Software: PyCharm
# @Description

import mxnet as mx
from mxnet.gluon import nn, rnn


class BiRnn(nn.Block):
    def __init__(self, vocab_size, num_hiddens, embed_size, num_layers, **kwargs):
        super(BiRnn, self).__init__(**kwargs)
        self.embedding = nn.Embedding(input_dim=vocab_size, output_dim=embed_size)
        self.encoder = rnn.LSTM(hidden_size=num_hiddens,
                                num_layers=num_layers,
                                bidirectional=True,
                                input_size=embed_size)
        self.decoder = nn.Dense(2)

    def forward(self, inputs):
        # inputs 形状是（批量大小，词数），因为 LSTM 需要将序列作为第一维，
        # 所以将输入转置后再提取词特征，输出形状为（词数，批量大小，词向量长度）。
        embedding = self.embedding(inputs.T)
        # states 形状是（词数，批量大小，2* 隐藏单元个数）。
        states = self.encoder(embedding)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入。
        # 它的形状为（批量大小，2* 隐藏单元个数）。
        encoding = mx.nd.concat(*[states[0], states[-1]])
        output = self.decoder(encoding)
        return output
