import random
import os
import copy

import numpy
import librosa
import chainer
from chainer import configuration
from chainer import link


class mu_law(object):
    def __init__(self, mu=256, int_type=numpy.uint8, float_type=numpy.float32):
        self.mu = mu
        self.int_type = int_type
        self.float_type = float_type

    def transform(self, x):
        x = x.astype(self.float_type)
        y = numpy.sign(x) * numpy.log(1 + self.mu * numpy.abs(x)) / \
            numpy.log(1 + self.mu)
        y = numpy.digitize(y, 2 * numpy.arange(self.mu) / self.mu - 1) - 1
        return y.astype(self.int_type)

    def itransform(self, y):
        y = y.astype(self.float_type)
        y = 2 * y / self.mu - 1
        x = numpy.sign(y) / self.mu * ((self.mu) ** numpy.abs(y) - 1)
        return x.astype(self.float_type)


class Preprocess(object):
    def __init__(self, sr, n_fft, hop_length, n_mels, mu, top_db,
                 length, dataset, speaker_dic, use_logistic, random=True):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.mu = mu
        self.mu_law = mu_law(mu)
        self.top_db = top_db
        if length is None:
            self.length = None
        else:
            self.length = length + 1
        self.dataset = dataset
        self.speaker_dic = speaker_dic
        self.use_logistic = use_logistic
        self.random = random

    def __call__(self, path):
        # load data
        raw = self.read_file(path)
        raw, _ = librosa.effects.trim(raw, self.top_db)
        raw /= numpy.abs(raw).max()

        if not self.use_logistic:
            quantized = self.mu_law.transform(raw)

        if self.length is not None:
            if len(raw) <= self.length:
                # padding
                pad = self.length-len(raw)
                raw = numpy.concatenate(
                    (raw, numpy.zeros(pad, dtype=numpy.float32)))
                if not self.use_logistic:
                    quantized = numpy.concatenate(
                        (quantized,
                            self.mu // 2 * numpy.ones(pad, dtype=numpy.int32)))
            else:
                # triming
                if self.random:
                    start = random.randint(0, len(raw) - self.length-1)
                    raw = raw[start:start + self.length]
                    if not self.use_logistic:
                        quantized = quantized[start:start + self.length]
                else:
                    raw = raw[:self.length]
                    if not self.use_logistic:
                        quantized = quantized[:self.length]

        # make mel=spectrogram
        spectrogram = librosa.feature.melspectrogram(
            raw, self.sr, n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=self.n_mels)
        spectrogram = librosa.amplitude_to_db(
            spectrogram, ref=numpy.max)
        spectrogram += 40
        spectrogram /= 40

        # expand dimension
        if self.use_logistic:
            # expand channel
            raw = numpy.expand_dims(raw, 0)

            # expand height
            raw = numpy.expand_dims(raw, -1)
        else:
            one_hot = numpy.identity(self.mu)[quantized].astype(numpy.float32)
            one_hot = numpy.expand_dims(one_hot.T, 2)
            quantized = numpy.expand_dims(quantized.astype(numpy.int32), 1)
        spectrogram = numpy.expand_dims(spectrogram, 2).astype(numpy.float32)

        # get speaker-id
        if self.speaker_dic is None:
            speaker = [-1]
        else:
            if self.dataset == 'VCTK':
                speaker = self.speaker_dic[
                    os.path.basename(os.path.dirname(path))]
            elif self.dataset == 'ARCTIC':
                speaker = self.speaker_dic[
                    os.path.basename(os.path.dirname(os.path.dirname(path)))]
            speaker = numpy.int32(speaker)
        if self.use_logistic:
            return raw[:, :-1], speaker, spectrogram[:, :-1], raw[:, 1:]
        else:
            return one_hot[:, :-1], speaker, spectrogram[:, :-1], quantized[1:]

    def read_file(self, path):
        x, sr = librosa.core.load(path, self.sr, res_type='kaiser_fast')
        return x


class ExponentialMovingAverage(link.Chain):

    def __init__(self, target, decay=0.999):
        super(ExponentialMovingAverage, self).__init__()
        self.decay = decay
        with self.init_scope():
            self.target = target
            self.ema = copy.deepcopy(target)

    def __call__(self, *args, **kwargs):
        if configuration.config.train:
            ys = self.target(*args, **kwargs)
            for target_name, target_param in self.target.namedparams():
                for ema_name, ema_param in self.ema.namedparams():
                    if target_name == ema_name:
                        if not target_param.requires_grad \
                                or ema_param.array is None:
                            new_average = target_param.array
                        else:
                            new_average = self.decay * target_param.array + \
                                (1 - self.decay) * ema_param.array
                        ema_param.array = new_average
        else:
            ys = self.ema(*args, **kwargs)
        return ys
