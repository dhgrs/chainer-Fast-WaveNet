import glob
import os
import numpy
import chainer

from WaveNet import WaveNet
from utils import Preprocess
import opt

# set data
if opt.dataset == 'VCTK':
    speakers = glob.glob(os.path.join(opt.root, 'wav48/*'))
    path = os.path.join(opt.root, 'wav48/p225/p225_001.wav')
elif opt.dataset == 'ARCTIC':
    speakers = glob.glob(os.path.join(opt.root, '*'))
    path = os.path.join(opt.root, 'cmu_us_bdl_arctic/wav/arctic_a0001.wav')

n_speaker = len(speakers)
speaker_dic = {
    os.path.basename(speaker): i for i, speaker in enumerate(speakers)}

# preprocess
n = 1
inputs = Preprocess(
    opt.data_format, opt.sr, opt.n_fft, opt.hop_length, opt.n_mels,
    opt.quantize, opt.top_db, None, opt.dataset, speaker_dic, opt.use_logistic,
    False)(path)

if opt.use_logistic:
    raw, global_cond, local_cond, t = inputs
    x = numpy.expand_dims(raw, 0)
    x = chainer.Variable(x)

else:
    one_hot, global_cond, local_cond, t = inputs
    x = numpy.expand_dims(one_hot, 0)
    x = chainer.Variable(x)

if not opt.global_conditioned:
    global_cond = None
    n_speaker = None

global_cond = numpy.expand_dims(global_cond, 0)
local_cond = numpy.expand_dims(local_cond, 0)

# make model
model1 = WaveNet(
    opt.n_loop, opt.n_layer, opt.filter_size, opt.quantize,
    opt.residual_channels, opt.dilated_channels, opt.skip_channels,
    opt.use_logistic, opt.global_conditioned, opt.local_conditioned,
    opt.n_mixture, opt.log_scale_min, n_speaker, opt.embed_dim,
    opt.n_mels, opt.upsample_factor, opt.use_deconv,
    opt.dropout_zero_rate)
model2 = model1.copy()

if opt.global_conditioned:
    global_cond = model1.embed_global_cond(global_cond)
if opt.local_conditioned:
    local_cond = model1.upsample_local_cond(local_cond)

model1.initialize(n, global_cond)
print(local_cond.shape, x.shape)
print('check fast generation and naive generation')
for i in range(opt.sr):
    with chainer.using_config('enable_backprop', False):
        with chainer.using_config('train', False):
            if opt.local_conditioned:
                out1 = model1.generate(
                    x[:, :, i:i+1], local_cond[:, :, i:i+1])
                out2 = model2(
                    x[:, :, :i+1], global_cond, local_cond[:, :, :i+1],
                    generating=True)
            else:
                out1 = model1.generate(
                    x[:, :, i:i+1], local_cond)
                out2 = model2(
                    x[:, :, :i+1], global_cond, local_cond, generating=True)
        print(
            '{}th sample, both of the values are same?:'.format(i),
            numpy.allclose(numpy.squeeze(out1.array),
                           numpy.squeeze(out2[:, :, -1:].array),
                           1e-3, 1e-5))
