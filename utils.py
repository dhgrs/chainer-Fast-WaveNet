# coding: UTF-8
import random
import os

import numpy as np
import librosa
from matplotlib import pyplot as plt


class mu_law(object):
    def __init__(self, mu=256, int_type=np.uint8, float_type=np.float32):
        self.mu = mu
        self.int_type = int_type
        self.float_type = float_type

    def transform(self, x):
        x = x.astype(self.float_type)
        y = np.sign(x) * np.log(1 + self.mu * np.abs(x)) / np.log(1 + self.mu)
        y = np.digitize(y, 2 * np.arange(self.mu) / self.mu - 1) - 1
        return y.astype(self.int_type)

    def itransform(self, y):
        y = y.astype(self.float_type)
        y = 2 * y / self.mu - 1
        x = np.sign(y) / self.mu * ((self.mu) ** np.abs(y) - 1)
        return x.astype(self.float_type)


class Preprocess(object):
    def __init__(self, sr, mu, length, random):
        self.sr = sr
        self.mu = mu
        self.mu_law = mu_law(mu)
        self.length = length + 1
        self.random = random

    def __call__(self, path):
        # load data
        npy_path = path.replace('.flac',
                                '_{}_{}.npy'.format(self.sr, self.mu+1))
        if os.path.exists(npy_path):
            t = np.load(npy_path)
        else:
            x = self.read_file(path)
            t = self.mu_law.transform(x)
            np.save(npy_path, t)

        # triming
        if self.random:
            start = random.randint(0, len(t) - self.length-1)
            t = t[start:start + self.length]
        else:
            t = t[:self.length]

        # expand dimension
        y = np.identity(self.mu + 1)[t].astype(np.float32)
        y = np.expand_dims(y.T, 2)
        t = np.expand_dims(t.astype(np.int32), 1)
        return y[:, :-1, :], t[1:, :]

    def read_file(self, path):
        x, sr = librosa.core.load(path, self.sr, res_type='kaiser_fast')
        return x
