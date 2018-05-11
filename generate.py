import glob
import os
import argparse

import numpy
import librosa
import chainer

from WaveNet import WaveNet
from utils import mu_law
from utils import Preprocess
from utils import ExponentialMovingAverage
import params

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', help='input file')
parser.add_argument('--output', '-o', default='result.wav', help='output file')
parser.add_argument('--model', '-m', help='snapshot of trained model')
parser.add_argument('--speaker', '-s', default=None,
                    help='name of speaker. if this is None,'
                         'input speaker is used.')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

# set data
if params.dataset == 'VCTK':
    speakers = glob.glob(os.path.join(params.root, 'wav48/*'))
elif params.dataset == 'ARCTIC':
    speakers = glob.glob(os.path.join(params.root, '*'))
path = args.input

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
    raw, speaker, local_cond, t = inputs
    raw = numpy.expand_dims(raw, 0)
    x = chainer.Variable(raw)

else:
    one_hot, global_cond, local_cond, t = inputs
    one_hot = numpy.expand_dims(one_hot, 0)
    x = chainer.Variable(one_hot)

if not params.global_conditioned:
    global_cond = None
    n_speaker = None

global_cond = numpy.expand_dims(global_cond, 0)
local_cond = numpy.expand_dims(local_cond, 0)

# make model
model = WaveNet(
    params.n_loop, params.n_layer, params.filter_size, params.quantize,
    params.residual_channels, params.dilated_channels, params.skip_channels,
    params.use_logistic, params.global_conditioned, params.local_conditioned,
    params.n_mixture, params.log_scale_min, n_speaker, params.embed_dim,
    params.n_mels, params.upsample_factor, params.use_deconv,
    params.dropout_zero_rate)

if params.ema_mu < 1:
    if params.use_ema:
        chainer.serializers.load_npz(
            args.model, model, 'updater/model:main/predictor/ema/')
    else:
        chainer.serializers.load_npz(
            args.model, model, 'updater/model:main/predictor/target/')
else:
    chainer.serializers.load_npz(
        args.model, model, 'updater/model:main/predictor/')

if args.gpu >= 0:
    use_gpu = True
    chainer.cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()
else:
    use_gpu = False

# forward
if use_gpu:
    x.array = chainer.cuda.to_gpu(x.array, device=args.gpu)
    if params.local_conditioned:
        local_cond = chainer.cuda.to_gpu(local_cond, device=args.gpu)
    if params.global_conditioned:
        global_cond = chainer.cuda.to_gpu(global_cond, device=args.gpu)
if params.local_conditioned:
    local_cond = model.upsample_local_cond(local_cond)
if params.global_conditioned:
    global_cond = model.embed(global_cond)
x = x[:, :, 0:1]
model.initialize(1, None)
output = model.xp.zeros(local_cond.shape[2])

for i in range(len(output) - 1):
    with chainer.using_config('enable_backprop', False):
        with chainer.using_config('train', params.apply_dropout):
            out = model.generate(x, local_cond[:, :, i:i+1]).array
    if params.use_logistic:
        nr_mix = out.shape[1] // 3

        logit_probs = out[:, :nr_mix]
        means = out[:, nr_mix:2 * nr_mix]
        log_scales = out[:, 2 * nr_mix:3 * nr_mix]
        log_scales = model.xp.maximum(log_scales, params.log_scale_min)

        # generate uniform
        rand = model.xp.random.uniform(0, 1, log_scales.shape)

        # convert into logistic
        rand = means + model.xp.exp(log_scales) * \
            (model.xp.log(rand) - model.xp.log(1 - rand))

        if params.sample_from_mixture:
            # generate uniform
            prob = model.xp.random.uniform(0, 1, logit_probs.shape)

            # apply softmax
            prob = logit_probs - model.xp.log(-model.xp.log(prob))

            # sample
            argmax = model.xp.eye(nr_mix)[prob.argmax(axis=1)]
            rand = rand * argmax.transpose((0, 3, 1, 2))

        else:
            # calculate mixture of logistic
            rand = chainer.functions.softmax(logit_probs).array * rand

        rand = model.xp.sum(rand, axis=1)
        value = model.xp.squeeze(rand.astype(model.xp.float32))
        value /= 127.5
        x.array[:] = value
    else:
        value = model.xp.random.choice(
            params.quantize, size=1,
            p=chainer.functions.softmax(out).array[0, :, 0, 0])
        zeros = model.xp.zeros_like(x.array)
        zeros[:, value, :, :] = 1
        x = chainer.Variable(zeros)
    output[i] = value

if use_gpu:
        output = chainer.cuda.to_cpu(output)
if params.use_logistic:
    wave = output
else:
    wave = mu_law(params.quantize).itransform(output)
librosa.output.write_wav(args.output, wave, params.sr)
