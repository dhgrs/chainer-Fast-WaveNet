import random
import copy

import numpy
import librosa
from chainer import configuration
from chainer import link


class MuLaw(object):
    def __init__(self, mu=256, int_type=numpy.int32, float_type=numpy.float32):
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
    def __init__(self, sr, n_fft, hop_length, n_mels, top_db, input_dim,
                 quantize, length, use_logistic):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.top_db = top_db
        if input_dim == 1:
            self.mu_law_input = False
        else:
            self.mu_law_input = True
            self.mu_law = MuLaw(quantize)
            self.quantize = quantize
        if length is None:
            self.length = None
        else:
            self.length = length + 1
        self.use_logistic = use_logistic
        if self.mu_law_input or not self.use_logistic:
            self.mu_law = MuLaw(quantize)
            self.quantize = quantize

    def __call__(self, path):
        # load data(trim and normalize)
        raw, _ = librosa.load(path, self.sr)
        raw, _ = librosa.effects.trim(raw, self.top_db)
        raw /= numpy.abs(raw).max()
        raw = raw.astype(numpy.float32)

        # mu-law transform
        if self.mu_law_input or not self.use_logistic:
            quantized = self.mu_law.transform(raw)

        # padding/triming
        if self.length is not None:
            if len(raw) <= self.length:
                # padding
                pad = self.length - len(raw)
                raw = numpy.concatenate(
                    (raw, numpy.zeros(pad, dtype=numpy.float32)))
                if self.mu_law_input or not self.use_logistic:
                    quantized = numpy.concatenate(
                        (quantized, self.quantize // 2 * numpy.ones(pad)))
                    quantized = quantized.astype(numpy.int32)
            else:
                # triming
                start = random.randint(0, len(raw) - self.length - 1)
                raw = raw[start:start + self.length]
                if self.mu_law_input or not self.use_logistic:
                    quantized = quantized[start:start + self.length]

        # make mel spectrogram
        spectrogram = librosa.feature.melspectrogram(
            raw, self.sr, n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=self.n_mels)
        spectrogram = librosa.power_to_db(
            spectrogram, ref=numpy.max)
        spectrogram += 40
        spectrogram /= 40
        if self.length is not None:
            spectrogram = spectrogram[:, :self.length // self.hop_length]
        spectrogram = spectrogram.astype(numpy.float32)

        # expand dimensions
        if self.mu_law_input:
            one_hot = numpy.identity(
                self.quantize, dtype=numpy.float32)[quantized]
            one_hot = numpy.expand_dims(one_hot.T, 2)
        raw = numpy.expand_dims(raw, 0)  # expand channel
        raw = numpy.expand_dims(raw, -1)  # expand height
        spectrogram = numpy.expand_dims(spectrogram, 2)
        if not self.use_logistic:
            quantized = numpy.expand_dims(quantized, 1)

        # return
        inputs = ()
        if self.mu_law_input:
            inputs += (one_hot[:, :-1],)
        else:
            inputs += (raw[:, :-1],)
        inputs += (spectrogram,)
        if self.use_logistic:
            inputs += (raw[:, 1:],)
        else:
            inputs += (quantized[1:],)
        return inputs


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
            xp = chainer.cuda.get_array_module(ys)
            if xp != numpy:
                xp.cuda.Device(ys.array.device).use()
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
