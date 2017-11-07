import sys

import numpy as np
import librosa
import chainer
import chainer.functions as F


import models
from utils import mu_law
import opt

model = models.WaveNet(opt.n_loop, opt.n_layer, opt.n_filter, opt.mu,
                       opt.n_channel1, opt.n_channel2, opt.n_channel3)
chainer.serializers.load_npz(sys.argv[1], model)

n = 1
model.initialize(n)
x = chainer.Variable(
    np.zeros(n * opt.mu, dtype=np.float32).reshape((n, opt.mu, 1, 1)))
x.data[:, opt.mu//2, :, :] = 1

output = np.zeros(opt.sr * int(sys.argv[2]))

for i in range(opt.sr * int(sys.argv[2])):
    with chainer.using_config('enable_backprop', False):
        out = model.generate(x)
    zeros = np.zeros_like(x.data)
    # value = out.data.argmax(axis=1)[0, 0, 0]
    value = np.random.choice(opt.mu, p=F.softmax(out).data[0, :, 0, 0])
    output[i] = value
    zeros[:, value, :, :] = 1
    x = chainer.Variable(zeros)

itransform = mu_law(opt.mu).itransform
wave = itransform(output)
np.save('result.npy', wave)
librosa.output.write_wav('result.wav', wave, opt.sr)
