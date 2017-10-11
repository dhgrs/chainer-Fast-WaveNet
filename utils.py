# coding: UTF-8
import random

import numpy as np
import librosa


class mu_law(object):
    def __init__(self, mu=255, int_type=np.int32, float_type=np.float32):
        self.mu = mu
        self.int_type = int_type
        self.float_type = float_type

    def transform(self, x):
        x = x.astype(self.float_type)
        y = np.sign(x) * np.log(1 + self.mu * np.abs(x)) / np.log(1 + self.mu)
        y = np.digitize(y, 2 * np.arange(1 + self.mu) / self.mu - 1) - 1
        y = np.identity(self.mu + 1)[y.astype(self.int_type)]
        return y.astype(self.int_type)

    def itransform(self, y):
        y = y.astype(self.float_type)
        y = 2 * y / self.mu - 1
        x = np.sign(y) / self.mu * ((1 + self.mu) ** np.abs(y) - 1)
        return x.astype(self.float_type)


class Preprocess(object):
    def __init__(self, sr, mu, length, random):
        self.sr = sr
        self.mu = mu
        self.mu_law = mu_law(mu)
        self.length = length + 1
        self.random = random

    def __call__(self, path):
        x = self.read_file(path)
        if self.random:
            start = random.randint(0, len(x) - self.length-1)
            x = x[start:start + self.length]
        else:
            x = x[:self.length]
        y = self.mu_law.transform(x)
        y = np.expand_dims(y.T, 1)
        return y[:, :, :-1].astype(mu_law.float_type), y[:, :, 1:]

    def read_file(self, path):
        x, sr = librosa.core.load(path, self.sr, res_type='kaiser_fast')
        return x
