import chainer
import chainer.links as L
import chainer.functions as F


def gated(x):
    n_channel = x.shape[1]
    return F.tanh(x[:, :n_channel // 2]) * F.sigmoid(x[:, n_channel // 2:])


class ResidualBlock(chainer.Chain):
    def __init__(self, dilation, n_channel1=32, n_channel2=16):
        super(ResidualBlock, self).__init__()
        with self.init_scope():
            self.conv = L.DilatedConvolution2D(n_channel2, n_channel1 * 2,
                                               ksize=(2, 1), pad=(dilation, 0),
                                               dilate=(dilation, 1))
            self.proj = L.Convolution2D(n_channel1, n_channel2, 1)

    def __call__(self, x):
        length = x.shape[2]
        # Dilated Conv
        x = self.conv(x)
        x = x[:, :, :length, :]

        # Gated activation units
        z = gated(x)

        # Projection
        z = self.proj(z)
        return z


class ResidualNet(chainer.ChainList):
    def __init__(self, n_loop, n_layer, n_filter,
                 n_channel1=32, n_channel2=16):
        super(ResidualNet, self).__init__()
        dilations = [
            n_filter ** i for j in range(n_loop) for i in range(n_layer)]
        for i, dilation in enumerate(dilations):
            self.add_link(ResidualBlock(dilation, n_channel1, n_channel2))

    def __call__(self, x):
        for i, func in enumerate(self.children()):
            a = x
            x = func(x)
            if i == 0:
                skip_connections = x
            else:
                skip_connections += x
            x = x + a
        return skip_connections


class WaveNet(chainer.Chain):
    def __init__(self, n_loop, n_layer, n_filter, quantize=256,
                 n_channel1=32, n_channel2=16, n_channel3=512):
        super(WaveNet, self).__init__()
        with self.init_scope():
            self.caus = L.Convolution2D(None, n_channel2, (2, 1), pad=(1, 0))
            self.resb = ResidualNet(n_loop, n_layer, n_filter,
                                    n_channel1, n_channel2)
            self.proj1 = L.Convolution2D(n_channel2, n_channel3, 1)
            self.proj2 = L.Convolution2D(n_channel3, quantize, 1)
        self.n_layer = n_layer

    def __call__(self, x, t=None):
        # Causal Conv
        length = x.shape[2]
        x = self.caus(x)
        x = x[:, :, :length, :]

        # Residual & Skip-conenection
        z = F.relu(self.resb(x))

        # Output
        z = F.relu(self.proj1(z))
        y = self.proj2(z)
        if chainer.config.train:
            loss = F.softmax_cross_entropy(y, t)
            accuracy = F.accuracy(y, t)
            chainer.report({'loss': loss, 'accuracy': accuracy}, self)
            return loss
        else:
            return y
