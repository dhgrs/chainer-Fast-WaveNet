# parameters of training
batchsize = 4
lr = 1e-4
ema_mu = 0.9999
dropout_zero_rate = 0.
trigger = (1000000, 'iteration')
evaluate_interval = (1, 'epoch')
snapshot_interval = (10000, 'iteration')
report_interval = (100, 'iteration')

# parameters of dataset
root = '../VCTK-Corpus/'
dataset = 'VCTK'

# parametars of mel-spectrogram
sr = 24000
n_fft = 1024
hop_length = 256
n_mels = 128

# parameters of preprocessing
quantize = 2 ** 16
top_db = 20
length = 7680

# parameters of WaveNet
global_conditioned = False
local_conditioned = True
upsample_factor = hop_length
use_deconv = True
use_logistic = True
n_mixture = 10 * 3
log_scale_min = -40
n_loop = 3
n_layer = 10
filter_size = 3
residual_channels = 512
dilated_channels = 512
skip_channels = 256
embed_dim = 64

# parameters of generating
sample_from_mixture = True
use_ema = True
apply_dropout = False
