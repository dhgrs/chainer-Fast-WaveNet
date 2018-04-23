import argparse
import datetime
import os
import shutil
import glob

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass
import chainer
from chainer.training import extensions

from utils import Preprocess
from utils import ExponentialMovingAverage
from WaveNet import WaveNet
import opt


# use CPU or GPU
parser = argparse.ArgumentParser()
parser.add_argument('--gpus', '-g', type=int, default=[-1], nargs='+',
                    help='GPU IDs (negative value indicates CPU)')
parser.add_argument('--process', '-p', type=int, default=1,
                    help='Number of parallel processes')
parser.add_argument('--prefetch', '-f', type=int, default=1,
                    help='Number of prefetch samples')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the training from snapshot')
args = parser.parse_args()

if args.gpus != [-1]:
    chainer.cuda.set_max_workspace_size(2*512*1024*1024)
    chainer.backends.cuda.get_device
    chainer.global_config.autotune = True

# get speaker dictionary
if opt.dataset == 'VCTK':
    speakers = glob.glob(os.path.join(opt.root, 'wav48/*'))
elif opt.dataset == 'ARCTIC':
    speakers = glob.glob(os.path.join(opt.root, '*'))
if opt.global_conditioned:
    n_speaker = len(speakers)
    speaker_dic = {
        os.path.basename(speaker): i for i, speaker in enumerate(speakers)}
else:
    n_speaker = None
    speaker_dic = None

# get paths
if opt.dataset == 'VCTK':
    files = glob.glob(os.path.join(opt.root, 'wav48/*/*.wav'))
elif opt.dataset == 'ARCTIC':
    files = glob.glob(os.path.join(opt.root, '*/wav/*.wav'))
elif opt.dataset == 'MIR':
    files = glob.glob(os.path.join(opt.root, 'Wavfile/*.wav'))

preprocess = Preprocess(
    opt.data_format, opt.sr, opt.n_fft, opt.hop_length, opt.n_mels,
    opt.quantize, opt.top_db, opt.length, opt.dataset, speaker_dic,
    opt.use_logistic)

dataset = chainer.datasets.TransformDataset(files, preprocess)
train, valid = chainer.datasets.split_dataset_random(
    dataset, int(len(dataset) * 0.9))

# make directory of results
result = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
os.mkdir(result)
shutil.copy(__file__, os.path.join(result, __file__))
shutil.copy('utils.py', os.path.join(result, 'utils.py'))
shutil.copy('modules.py', os.path.join(result, 'modules.py'))
shutil.copy('opt.py', os.path.join(result, 'opt.py'))
shutil.copy('generate.py', os.path.join(result, 'generate.py'))
shutil.copy('fast_generation_test.py',
            os.path.join(result, 'fast_generation_test.py'))

# Model
wavenet = WaveNet(
    opt.n_loop, opt.n_layer, opt.filter_size, opt.quantize,
    opt.residual_channels, opt.dilated_channels, opt.skip_channels,
    opt.use_logistic, opt.global_conditioned, opt.local_conditioned,
    opt.n_mixture, opt.log_scale_min, n_speaker, opt.embed_dim,
    opt.n_mels, opt.upsample_factor, opt.use_deconv,
    opt.dropout_zero_rate)
if opt.ema_mu < 1:
    ema = ExponentialMovingAverage(wavenet, opt.ema_mu)
else:
    ema = wavenet

if opt.use_logistic:
    loss_fun = wavenet.calculate_logistic_loss
    acc_fun = None
    model = chainer.links.Classifier(ema, loss_fun, acc_fun)
    model.compute_accuracy = False
else:
    loss_fun = chainer.functions.softmax_cross_entropy
    acc_fun = chainer.functions.accuracy
    model = chainer.links.Classifier(ema, loss_fun, acc_fun)

# Optimizer
optimizer = chainer.optimizers.Adam(opt.lr/len(args.gpus))
optimizer.setup(model)

# Iterator
if args.process * args.prefetch > 1:
    train_iter = chainer.iterators.MultiprocessIterator(
        train, opt.batchsize,
        n_processes=args.process, n_prefetch=args.prefetch)
    valid_iter = chainer.iterators.MultiprocessIterator(
        valid, opt.batchsize//len(args.gpus), repeat=False, shuffle=False,
        n_processes=args.process, n_prefetch=args.prefetch)
else:
    train_iter = chainer.iterators.SerialIterator(train, opt.batchsize)
    valid_iter = chainer.iterators.SerialIterator(
        valid, opt.batchsize//len(args.gpus), repeat=False, shuffle=False)

# Updater
if args.gpus == [-1]:
    updater = chainer.training.StandardUpdater(train_iter, optimizer)
else:
    chainer.cuda.get_device_from_id(args.gpus[0]).use()
    names = ['main'] + list(range(len(args.gpus)-1))
    devices = {str(name): gpu for name, gpu in zip(names, args.gpus)}
    updater = chainer.training.ParallelUpdater(
        train_iter, optimizer, devices=devices)

# Trainer
trainer = chainer.training.Trainer(updater, opt.trigger, out=result)

# Extensions
trainer.extend(extensions.Evaluator(valid_iter, model, device=args.gpus[0]),
               trigger=opt.evaluate_interval)
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.snapshot(), trigger=opt.snapshot_interval)
trainer.extend(extensions.LogReport(trigger=opt.report_interval))
trainer.extend(extensions.PrintReport(
    ['epoch', 'iteration', 'main/loss', 'main/accuracy',
     'validation/main/loss', 'validation/main/accuracy']),
    trigger=opt.report_interval)
trainer.extend(extensions.PlotReport(
    ['main/loss', 'validation/main/loss'],
    'iteration', file_name='loss.png', trigger=opt.report_interval))
trainer.extend(extensions.PlotReport(
    ['main/accuracy', 'validation/main/accuracy'],
    'iteration', file_name='accuracy.png', trigger=opt.report_interval))
trainer.extend(extensions.ProgressBar(update_interval=1))

if args.resume:
    chainer.serializers.load_npz(args.resume, trainer)

# run
print('GPUs: {}'.format(*args.gpus))
print('# train: {}'.format(len(train)))
print('# valid: {}'.format(len(valid)))
print('# Minibatch-size: {}'.format(opt.batchsize))
print('# {}: {}'.format(opt.trigger[1], opt.trigger[0]))
print('')

trainer.run()
