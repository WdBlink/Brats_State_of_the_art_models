import os

import torch

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from unet3d.config import load_config
from tqdm import tqdm


from unet3d import utils
from datasets.hdf5 import get_brats_train_loaders
from unet3d.utils import get_logger


# Load and log experiment configuration
logger = get_logger('MixUp')
config = load_config()
logger.info(config)
loaders = get_brats_train_loaders(config)
train_loader = loaders['train']

def _split_training_batch(t):
    def _move_to_device(input):
        if isinstance(input, tuple) or isinstance(input, list):

            return tuple([_move_to_device(input[0]), input[1], _move_to_device(input[2])])
        else:
            return input.to(0, dtype=torch.float)

    t = _move_to_device(t)
    if len(t) == 2:
        input, target = t
    else:
        input, pid, target = t
    return input, pid, target

mixup_data = 0
for i, t in enumerate(tqdm(train_loader)):
    input, pid, target = _split_training_batch(t)
    logger.info(f'mixing up {pid}')
    mixup_data += input
mixup_data = mixup_data / 368
cpu = mixup_data.data.cpu()
mixup_data = cpu.numpy()

basePath = '/home/server/data/BraTS19/training'

path = os.path.join(basePath, "MixupData.nii.gz")
utils.save_nii(path, mixup_data, None, None)

img_3 = mixup_data[0, 1, :, :, mixup_data.shape(4)//2]
fig = plt.gcf()
fig.set_size_inches(5, 5)
plt.imshow(img_3)
plt.savefig(f'{basePath}/mixup.png')
plt.show()

logger.info('finished')