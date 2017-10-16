# data directories
train_dir = 'train/'
valid_dir = 'valid/'
data_format = 'flac'

# input parameter
batchsize = 4
sr = 8000
mu = 256
length = 8192 * 4
random = True

# network parameter
n_loop = 3
n_layer = 8
n_filter = 2
n_channel1 = 64
n_channel2 = 32
n_channel3 = 512

# learning parameter
lr = 1e-4

# trigger
epoch = 10
