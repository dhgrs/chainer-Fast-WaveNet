import chainer
import chainer.functions as F
import chainer.links as L


class UpsampleNet(chainer.ChainList):
    def __init__(self, channels, upscale_factors):
        super(UpsampleNet, self).__init__()
        for channel, factor in zip(channels, upscale_factors):
            self.add_link(L.Deconvolution2D(
                None, channel, (factor, 1), stride=(factor, 1), pad=0))

    def __call__(self, x):
        for link in self.children():
            x = F.relu(link(x))
        return x


class EncoderDecoderModel(chainer.Chain):
    def __init__(self, encoder, decoder, loss_fun, acc_fun):
        super(EncoderDecoderModel, self).__init__()
        with self.init_scope():
            self.encoder = encoder
            self.decoder = decoder
        self.loss_fun = loss_fun
        self.acc_fun = acc_fun

    def __call__(self, x, condition, t):
        condition = self.encoder(condition)
        y = self.decoder(x, condition)
        loss = self.loss_fun(y, t)
        if self.acc_fun is None:
            chainer.reporter.report({'loss': loss}, self)
        else:
            chainer.reporter.report({
                'loss': loss, 'accuracy': self.acc_fun(y, t)}, self)
        return loss
