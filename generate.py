import numpy as np
import chainer

import models
from utils import Preprocess
import opt

model = models.WaveNet(opt.n_loop, opt.n_layer, opt.n_filter, opt.mu,
                       opt.n_channel1, opt.n_channel2, opt.n_channel3)

n = 1
model.initialize(n)
x = chainer.Variable(
    np.arange(n * opt.mu, dtype=np.float32).reshape((n, opt.mu, 1, 1)))

for i in range(opt.sr * 10):
    x = model.generate(x)
    print(x.shape)
