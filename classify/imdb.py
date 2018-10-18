#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 18-10-15 下午5:38
# @Author  : guoxz
# @Site    : 
# @File    : imdb.py
# @Software: PyCharm
# @Description

import os
import gluonnlp as nlp
import codecs
from mxnet import gluon


@nlp.data.register(segment=['train', 'test', 'unsup'])
class MyIMDB(gluon.data.SimpleDataset):
    def __init__(self, segment='train', root='/opt/disk1/nlp/aclImdb'):
        self._segment = segment
        self._root = root
        super(MyIMDB, self).__init__(self._read_data())

    def _read_data(self):
        def read(txt_path):
            with codecs.open(txt_path, 'r') as f:
                lines = f.readlines()
            return ''.join([line + ' ' for line in lines])

        readline = lambda dir, file_path: read(os.path.join(dir, file_path))

        pos_dir = os.path.join(self._root, self._segment, 'pos')
        neg_dir = os.path.join(self._root, self._segment, 'neg')

        data = [list((readline(pos_dir, file_path), 1)) for file_path in os.listdir(pos_dir)] + [
            list((readline(neg_dir, file_path), 0)) for file_path in os.listdir(neg_dir)]

        return data


if __name__ == '__main__':
    train_data = MyIMDB(segment='train')
    print(train_data[0])
