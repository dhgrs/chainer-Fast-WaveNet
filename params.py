# parameters of training
batchsize = 4
lr = 1e-4
ema_mu = 0.9999
trigger = (1000000, 'iteration')
evaluate_interval = (1, 'epoch')
snapshot_interval = (10000, 'iteration')
report_interval = (100, 'iteration')

# parameters of dataset
root = '../LJSpeech-1.1/'
dataset_type = 'LJSpeech'
split_seed = 71

# parameters of preprocessing
sr = 24000
n_fft = 1024
hop_length = 256
n_mels = 80
top_db = 20
input_dim = 1
quantize = 2 ** 16
length = 7680
use_logistic = True

# parameters of Encoder(Deconvolution network)
channels = [128, 128]
upsample_factors = [16, 16]

# parameters of Decoder(WaveNet)
n_loop = 3
n_layer = 10
filter_size = 3
input_dim = input_dim
residual_channels = 512
dilated_channels = 512
skip_channels = 256
# quantize = quantize
# use_logistic = use_logistic
n_mixture = 10 * 3
log_scale_min = -40
condition_dim = channels[-1]
dropout_zero_rate = 0

# parameters of generating
use_ema = True
apply_dropout = False
