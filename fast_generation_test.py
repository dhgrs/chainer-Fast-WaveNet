import glob
import os
import numpy
import chainer

from WaveNet import WaveNet
from utils import Preprocess
import params

# set data
if params.dataset == 'VCTK':
    speakers = glob.glob(os.path.join(params.root, 'wav48/*'))
    path = os.path.join(params.root, 'wav48/p225/p225_001.wav')
elif params.dataset == 'ARCTIC':
    speakers = glob.glob(os.path.join(params.root, '*'))
    path = os.path.join(params.root, 'cmu_us_bdl_arctic/wav/arctic_a0001.wav')

n_speaker = len(speakers)
speaker_dic = {
    os.path.basename(speaker): i for i, speaker in enumerate(speakers)}

# preprocess
n = 1
inputs = Preprocess(
    params.sr, params.n_fft, params.hop_length, params.n_mels, params.quantize,
    params.top_db, None, params.dataset, speaker_dic, params.use_logistic,
    False)(path)

if params.use_logistic:
    raw, global_cond, local_cond, t = inputs
    x = numpy.expand_dims(raw, 0)
    x = chainer.Variable(x)

else:
    one_hot, global_cond, local_cond, t = inputs
    x = numpy.expand_dims(one_hot, 0)
    x = chainer.Variable(x)

if not params.global_conditioned:
    global_cond = None
    n_speaker = None

global_cond = numpy.expand_dims(global_cond, 0)
local_cond = numpy.expand_dims(local_cond, 0)

# make model
model1 = WaveNet(
    params.n_loop, params.n_layer, params.filter_size, params.quantize,
    params.residual_channels, params.dilated_channels, params.skip_channels,
    params.use_logistic, params.global_conditioned, params.local_conditioned,
    params.n_mixture, params.log_scale_min, n_speaker, params.embed_dim,
    params.n_mels, params.upsample_factor, params.use_deconv,
    params.dropout_zero_rate)
model2 = model1.copy()

if params.global_conditioned:
    global_cond = model1.embed_global_cond(global_cond)
if params.local_conditioned:
    local_cond = model1.upsample_local_cond(local_cond)

model1.initialize(n, global_cond)
print(local_cond.shape, x.shape)
print('check fast generation and naive generation')
for i in range(params.sr):
    with chainer.using_config('enable_backprop', False):
        with chainer.using_config('train', False):
            if params.local_conditioned:
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
