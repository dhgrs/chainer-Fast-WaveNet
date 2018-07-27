import argparse

import numpy
import librosa
import chainer
import tqdm

from WaveNet import WaveNet
from net import UpsampleNet
from utils import MuLaw
from utils import Preprocess
import params

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', help='input file')
parser.add_argument('--output', '-o', default='result.wav', help='output file')
parser.add_argument('--model', '-m', help='snapshot of trained model')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()
if args.gpu != [-1]:
    chainer.cuda.set_max_workspace_size(2 * 512 * 1024 * 1024)
    chainer.global_config.autotune = True

# set data
path = args.input

# preprocess
n = 1  # batchsize; now suporrts only 1
inputs = Preprocess(
    params.sr, params.n_fft, params.hop_length, params.n_mels, params.top_db,
    params.input_dim, params.quantize, None, params.use_logistic)(path)

_, condition, _ = inputs
x = numpy.zeros([n, params.input_dim, 1, 1], dtype=numpy.float32)
condition = numpy.expand_dims(condition, axis=0)

# make model
encoder = UpsampleNet(params.channels, params.upsample_factors)
decoder = WaveNet(
    params.n_loop, params.n_layer, params.filter_size, params.input_dim,
    params.residual_channels, params.dilated_channels, params.skip_channels,
    params.quantize, params.use_logistic, params.n_mixture,
    params.log_scale_min,
    params.condition_dim, params.dropout_zero_rate)

# load trained parameter
chainer.serializers.load_npz(
    args.model, encoder, 'updater/model:main/encoder/')
if params.ema_mu < 1:
    if params.use_ema:
        chainer.serializers.load_npz(
            args.model, decoder, 'updater/model:main/decoder/ema/')
    else:
        chainer.serializers.load_npz(
            args.model, decoder, 'updater/model:main/decoder/target/')
else:
    chainer.serializers.load_npz(
        args.model, decoder, 'updater/model:main/decoder/')

if args.gpu >= 0:
    use_gpu = True
    chainer.cuda.get_device_from_id(args.gpu).use()
else:
    use_gpu = False

# forward
if use_gpu:
    x = chainer.cuda.to_gpu(x, device=args.gpu)
    condition = chainer.cuda.to_gpu(condition, device=args.gpu)
    encoder.to_gpu(device=args.gpu)
    decoder.to_gpu(device=args.gpu)
x = chainer.Variable(x)
condition = chainer.Variable(condition)
condition = encoder(condition)
decoder.initialize(n)
output = decoder.xp.zeros(condition.shape[2])

for i in tqdm.tqdm(range(len(output) - 1)):
    with chainer.using_config('enable_backprop', False):
        with chainer.using_config('train', params.apply_dropout):
            out = decoder.generate(x, condition[:, :, i:i + 1]).array
    if params.use_logistic:
        nr_mix = out.shape[1] // 3

        logit_probs = out[:, :nr_mix]
        means = out[:, nr_mix:2 * nr_mix]
        log_scales = out[:, 2 * nr_mix:3 * nr_mix]
        log_scales = decoder.xp.maximum(log_scales, params.log_scale_min)

        # generate uniform
        rand = decoder.xp.random.uniform(0, 1, logit_probs.shape)

        # apply softmax
        prob = logit_probs - decoder.xp.log(-decoder.xp.log(rand))

        # sample
        argmax = prob.argmax(axis=1)
        means = means[:, argmax]
        log_scales = log_scales[:, argmax]

        # generate uniform
        rand = decoder.xp.random.uniform(0, 1, log_scales.shape)

        # convert into logistic
        rand = means + decoder.xp.exp(log_scales) * \
            (decoder.xp.log(rand) - decoder.xp.log(1 - rand))

        value = decoder.xp.squeeze(rand.astype(decoder.xp.float32))
        value /= 127.5
        x.array[:] = value
    else:
        value = decoder.xp.random.choice(
            params.quantize,
            p=chainer.functions.softmax(out).array[0, :, 0, 0])
        zeros = decoder.xp.zeros_like(x.array)
        zeros[:, value, :, :] = 1
        x = chainer.Variable(zeros)
    output[i] = value

if use_gpu:
    output = chainer.cuda.to_cpu(output)
if params.use_logistic:
    wave = output
else:
    wave = MuLaw(params.quantize).itransform(output)
librosa.output.write_wav(args.output, wave, params.sr)
