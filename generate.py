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
import opt

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
if opt.dataset == 'VCTK':
    speakers = glob.glob(os.path.join(opt.root, 'wav48/*'))
elif opt.dataset == 'ARCTIC':
    speakers = glob.glob(os.path.join(opt.root, '*'))
path = args.input

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
    raw, speaker, local_cond, t = inputs
    raw = numpy.expand_dims(raw, 0)
    x = chainer.Variable(raw)

else:
    one_hot, global_cond, local_cond, t = inputs
    one_hot = numpy.expand_dims(one_hot, 0)
    x = chainer.Variable(one_hot)

if not opt.global_conditioned:
    global_cond = None
    n_speaker = None

global_cond = numpy.expand_dims(global_cond, 0)
local_cond = numpy.expand_dims(local_cond, 0)

# make model
model = WaveNet(
    opt.n_loop, opt.n_layer, opt.filter_size, opt.quantize,
    opt.residual_channels, opt.dilated_channels, opt.skip_channels,
    opt.use_logistic, opt.global_conditioned, opt.local_conditioned,
    opt.n_mixture, opt.log_scale_min, n_speaker, opt.embed_dim,
    opt.n_mels, opt.upsample_factor, opt.use_deconv,
    opt.dropout_zero_rate)

if opt.ema_mu < 1:
    if opt.use_ema:
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
    if opt.local_conditioned:
        local_cond = chainer.cuda.to_gpu(local_cond, device=args.gpu)
    if opt.global_conditioned:
        global_cond = chainer.cuda.to_gpu(global_cond, device=args.gpu)
if opt.local_conditioned:
    local_cond = model.upsample_local_cond(local_cond)
if opt.global_conditioned:
    global_cond = model.embed(global_cond)
x = x[:, :, 0:1]
model.initialize(1, None)
output = model.xp.zeros(local_cond.shape[2])

for i in range(len(output) - 1):
    with chainer.using_config('enable_backprop', False):
        with chainer.using_config('train', opt.apply_dropout):
            out = model.generate(x, local_cond[:, :, i:i+1]).array
    if opt.use_logistic:
        nr_mix = out.shape[1] // 3

        logit_probs = out[:, :nr_mix]
        means = out[:, nr_mix:2 * nr_mix]
        log_scales = out[:, 2 * nr_mix:3 * nr_mix]
        log_scales = model.xp.maximum(log_scales, opt.log_scale_min)

        # generate uniform
        rand = model.xp.random.uniform(0, 1, log_scales.shape)

        # convert into logistic
        rand = means + model.xp.exp(log_scales) * \
            (model.xp.log(rand) - model.xp.log(1 - rand))

        if opt.sample_from_mixture:
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
        x.array[:] = value
    else:
        value = model.xp.random.choice(
            opt.quantize, size=1,
            p=chainer.functions.softmax(out).array[0, :, 0, 0])
        zeros = model.xp.zeros_like(x.array)
        zeros[:, value, :, :] = 1
        x = chainer.Variable(zeros)
    output[i] = value

if use_gpu:
        output = chainer.cuda.to_cpu(output)
if opt.use_logistic:
    wave = output
else:
    wave = mu_law(opt.quantize).itransform(output)
librosa.output.write_wav(args.output, wave, opt.sr)
