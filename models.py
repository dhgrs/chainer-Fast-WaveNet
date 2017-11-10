import chainer
import chainer.links as L
import chainer.functions as F


def gated(x, h=None):
    n_channel = x.shape[1]
    if h is not None:
        x += h
    return F.tanh(x[:, :n_channel // 2]) * F.sigmoid(x[:, n_channel // 2:])


class ResidualBlock(chainer.Chain):
    def __init__(self, dilation, n_channel1=32, n_channel2=16):
        super(ResidualBlock, self).__init__()
        with self.init_scope():
            self.conv = L.DilatedConvolution2D(n_channel2, n_channel1 * 2,
                                               ksize=(2, 1), pad=(dilation, 0),
                                               dilate=(dilation, 1))
            self.cond = L.DilatedConvolution2D(None, n_channel1 * 2,
                                               ksize=(2, 1), pad=(dilation, 0))
            self.proj = L.Convolution2D(n_channel1, n_channel2, 1)
        self.dilation = dilation
        self.n_channel2 = n_channel2

    def __call__(self, x, h=None):
        length = x.shape[2]
        # Dilated Conv
        x = self.conv(x)
        x = x[:, :, :length, :]
        if h is not None:
            h = self.cond(h)
            h = h[:, :, :length, :]

        # Gated activation units
        z = gated(x, h)

        # Projection
        z = self.proj(z)
        return z

    def initialize(self, n, cond=False):
        self.queue = chainer.Variable(
            self.xp.zeros((n, self.n_channel2, self.dilation + 1, 1),
                          dtype=self.xp.float32))
        self.conv.pad = (0, 0)
        if cond:
            self.cond = chainer.Variable(
                self.xp.zeros((n, self.n_channel2, self.dilation + 1, 1),
                              dtype=self.xp.float32))
            self.cond.pad = (0, 0)
        else:
            self.cond = None

    def pop(self):
        return self.__call__(self.queue, self.cond)

    def push(self, sample):
        self.queue = F.concat((self.queue[:, :, 1:, :], sample), axis=2)


class ResidualNet(chainer.ChainList):
    def __init__(self, n_loop, n_layer, n_filter,
                 n_channel1=32, n_channel2=16):
        super(ResidualNet, self).__init__()
        dilations = [
            n_filter ** i for j in range(n_loop) for i in range(n_layer)]
        for i, dilation in enumerate(dilations):
            self.add_link(ResidualBlock(dilation, n_channel1, n_channel2))

    def __call__(self, x, h=None):
        for i, func in enumerate(self.children()):
            a = x
            x = func(x, h)
            if i == 0:
                skip_connections = x
            else:
                skip_connections += x
            x = x + a
        return skip_connections

    def initialize(self, n, cond=False):
        for block in self.children():
            block.initialize(n, cond)

    def generate(self, x, h=None):
        sample = x
        for i, func in enumerate(self.children()):
            a = sample
            func.push(sample, h)
            sample = func.pop()
            if i == 0:
                skip_connections = sample
            else:
                skip_connections += sample
            sample = sample + a
        return skip_connections


class WaveNet(chainer.Chain):
    def __init__(self, n_loop, n_layer, n_filter, quantize=256,
                 n_channel1=32, n_channel2=16, n_channel3=512):
        super(WaveNet, self).__init__()
        with self.init_scope():
            self.caus = L.Convolution2D(
                quantize, n_channel2, (2, 1), pad=(1, 0))
            self.resb = ResidualNet(
                n_loop, n_layer, n_filter, n_channel1, n_channel2)
            self.proj1 = L.Convolution2D(n_channel2, n_channel3, 1)
            self.proj2 = L.Convolution2D(n_channel3, quantize, 1)
        self.n_layer = n_layer
        self.quantize = quantize
        self.n_channel2 = n_channel2
        self.n_channel3 = n_channel3

    def __call__(self, x, t=None, h=None):
        # Causal Conv
        length = x.shape[2]
        x = self.caus(x)
        x = x[:, :, :length, :]

        # Residual & Skip-conenection
        z = F.relu(self.resb(x, h))

        # Output
        z = F.relu(self.proj1(z))
        y = self.proj2(z)
        if t is None:
            return y

        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        chainer.report({'loss': loss, 'accuracy': accuracy}, self)
        return loss

    def initialize(self, n, cond=None):
        self.resb.initialize(n, cond)
        self.caus.pad = (0, 0)
        self.queue1 = chainer.Variable(
            self.xp.zeros((n, self.quantize, 2, 1), dtype=self.xp.float32))
        # self.queue1.data[:, self.quantize//2, :, :] = 1
        self.queue2 = chainer.Variable(
            self.xp.zeros((n, self.n_channel2, 1, 1), dtype=self.xp.float32))
        self.queue3 = chainer.Variable(
            self.xp.zeros((n, self.n_channel3, 1, 1), dtype=self.xp.float32))

    def generate(self, x, h=None):
        self.queue1 = F.concat((self.queue1[:, :, 1:, :], x), axis=2)
        x = self.caus(self.queue1)
        x = F.relu(self.resb.generate(x, h))
        self.queue2 = F.concat((self.queue2[:, :, 1:, :], x), axis=2)
        x = F.relu(self.proj1(self.queue2))
        self.queue3 = F.concat((self.queue3[:, :, 1:, :], x), axis=2)
        x = self.proj2(self.queue3)
        return x
