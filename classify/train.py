#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 18-10-16 下午3:16
# @Author  : guoxz
# @Site    : 
# @File    : train.py
# @Software: PyCharm
# @Description

from mxnet import autograd


def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1).reshape(y.shape) == y.astype('float32')).mean().asscalar()


def eval_data(data_iter, net, ctx):
    acc = 0
    for X, y in data_iter:
        X = X.as_in_context(ctx)
        y = y.as_in_context(ctx)

        y_hat = net(X)
        acc += accuracy(y_hat, y)
    return acc / len(data_iter)


def train(train_iter, test_iter, loss, net, trainer, num_epochs, ctx, disp_batch=20):
    for epoch in range(num_epochs):

        train_acc_sum = 0
        train_l_sum = 0

        # batch_acc_sum = 0
        # batch_l_sum = 0

        for i, (X, y) in enumerate(train_iter):
            X = X.as_in_context(ctx)
            y = y.as_in_context(ctx)

            batch_size = len(X)

            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(batch_size)

            train_acc_sum += accuracy(y_hat, y)
            train_l_sum += l.sum().asscalar()

            if (i + 1) % 10 == 0:
                print('batch %d' % (i + 1,))

        train_acc_sum /= len(train_iter)
        train_l_sum /= len(train_iter)

        test_acc = eval_data(test_iter, net, ctx)

        print(
            'epoch %d, train loss %.4f, train acc %.4f, test acc %.4f' % (epoch, train_l_sum, train_acc_sum, test_acc))
