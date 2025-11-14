# minimal essential config for U-Net

seed = 42
device = 'cuda'

# data
batch_size = 1
tile_size = 256
overlap = 32

# model
in_channels = 1
out_channels = 2
base_filters = 64

# training
epochs = 50
lr = 0.001
momentum = 0.99
weight_decay = 1e-4
use_weight_map = True

# loss & weight map
loss_type = 'weighted_cross_entropy'
optimizer_type = 'SGD'

# checkpoints/log
save_dir = './checkpoints'
log_dir = './logs'
save_every = 5
