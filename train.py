# coding: UTF-8
import argparse
import datetime
import os
import shutil
import glob

import matplotlib
matplotlib.use('Agg')
import chainer
from chainer.training import extensions

import models
from utils import Preprocess
import opt

parser = argparse.ArgumentParser(description='WaveNet')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--GPU', '-G', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--process', '-p', type=int, default=1,
                    help='Number of parallel processes')
parser.add_argument('--prefetch', '-f', type=int, default=1,
                    help='Number of prefetch samples')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the training from snapshot')


args = parser.parse_args()

# load data
preprocess = Preprocess(
    opt.data_format, opt.sr, opt.mu, opt.length, opt.random)
train = chainer.datasets.TransformDataset(glob.glob(
    os.path.join(opt.train_dir, '*.{}'.format(opt.data_format))), preprocess)
valid = chainer.datasets.TransformDataset(glob.glob(os.path.join(
    opt.valid_dir, '*.{}'.format(opt.data_format))), preprocess)

# make directory of results
result = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
os.mkdir(result)
shutil.copy(__file__, result + '/' + __file__)
shutil.copy('models.py', result + '/' + 'models.py')
shutil.copy('utils.py', result + '/' + 'utils.py')
shutil.copy('opt.py', result + '/' + 'opt.py')
shutil.copy('generate.py', result + '/' + 'generate.py')


# Model
gpu = max(args.gpu, args.GPU)
if args.gpu >= 0 and args.GPU >= 0:
    chainer.cuda.get_device_from_id(gpu).use()
model = models.WaveNet(opt.n_loop, opt.n_layer, opt.n_filter, opt.mu,
                       opt.n_channel1, opt.n_channel2, opt.n_channel3)

# Optimizer
optimizer = chainer.optimizers.Adam(opt.lr)
optimizer.setup(model)


# Iterator
if args.process * args.prefetch > 1:
    train_iter = chainer.iterators.MultiprocessIterator(
        train, opt.batchsize,
        n_processes=args.process, n_prefetch=args.prefetch)
    valid_iter = chainer.iterators.MultiprocessIterator(
        valid, opt.batchsize, repeat=False, shuffle=False,
        n_processes=args.process, n_prefetch=args.prefetch)
else:
    train_iter = chainer.iterators.SerialIterator(train, opt.batchsize)
    valid_iter = chainer.iterators.SerialIterator(valid, opt.batchsize,
                                                  repeat=False, shuffle=False)

# Updater
if args.gpu >= 0 and args.GPU >= 0:
    chainer.cuda.get_device_from_id(args.gpu).use()
    updater = chainer.training.ParallelUpdater(
        train_iter, optimizer,
        devices={'main': gpu, 'second': min(args.gpu, args.GPU)})
elif args.gpu >= 0 or args.GPU >= 0:
    chainer.cuda.get_device_from_id(args.gpu).use()
    updater = chainer.training.StandardUpdater(
        train_iter, optimizer, device=gpu)
else:
    updater = chainer.training.StandardUpdater(train_iter, optimizer)

# Trainer
trainer = chainer.training.Trainer(updater, opt.trigger, out=result)

# Extensions
trainer.extend(extensions.Evaluator(valid_iter, model, device=gpu),
               trigger=(1000, 'iteration'))
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.ExponentialShift('alpha', 0.1),
               trigger=(100000, 'iteration'))
trainer.extend(extensions.observe_value(
    'alpha', lambda trainer: trainer.updater.get_optimizer('main').alpha),
    trigger=(5, 'iteration'))
trainer.extend(extensions.observe_lr(), trigger=(5, 'iteration'))
trainer.extend(extensions.snapshot_object(model, 'model{.updater.iteration}'),
               trigger=(1000, 'iteration'))
trainer.extend(extensions.LogReport(trigger=(5, 'iteration')))
trainer.extend(extensions.PrintReport(
    ['epoch', 'iteration', 'main/loss', 'main/accuracy',
     'validation/main/loss', 'validation/main/accuracy']),
    trigger=(5, 'iteration'))
trainer.extend(extensions.PlotReport(
    ['main/loss', 'validation/main/loss'],
    'iteration', file_name='loss.png', trigger=(5, 'iteration')))
trainer.extend(extensions.PlotReport(
    ['main/accuracy', 'validation/main/accuracy'],
    'iteration', file_name='accuracy.png', trigger=(5, 'iteration')))
trainer.extend(extensions.PlotReport(
    ['lr', 'alpha'],
    'iteration', file_name='lr.png', trigger=(5, 'iteration')))
trainer.extend(extensions.ProgressBar(update_interval=5))

if args.resume:
    chainer.serializers.load_npz(args.resume, trainer)

# run
print('GPU1: {}'.format(args.gpu))
print('GPU2: {}'.format(args.GPU))
print('# Minibatch-size: {}'.format(opt.batchsize))
print('# {}: {}'.format(opt.trigger[1], opt.trigger[0]))
print('')

trainer.run()
