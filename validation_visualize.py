from unet3d.config import load_config
import os
import torch
import torch.nn as nn
import datetime
from datasets.hdf5 import BratsDataset
from unet3d.model import get_model
from unet3d import utils
from tensorboardX import SummaryWriter
from visualization import board_add_images, board_add_image


def get_job_name():
    now = '{:%Y-%m-%d.%H:%M}'.format(datetime.datetime.now())
    return "%s_model" % (now)

logger = utils.get_logger('UNet3DPredictor')

# Load and log experiment configuration
config = load_config()

# Load model state
model = get_model(config)
model_path = config['trainer']['test_model']
logger.info(f'Loading model from {model_path}...')
utils.load_checkpoint(model_path, model)

logger.info(f"Sending the model to '{config['device']}'")
model = model.to('cuda:0')

predictionsBasePath = config['loaders']['pred_path']
BRATS_VAL_PATH = config['loaders']['test_path']

challengeValset = BratsDataset(BRATS_VAL_PATH[0], mode="validation", hasMasks=False, returnOffsets=True)
challengeValloader = torch.utils.data.DataLoader(challengeValset, batch_size=1, shuffle=False, pin_memory=True, num_workers=1)

writer = SummaryWriter(logdir=os.path.join(predictionsBasePath[0], get_job_name()))


def makePredictions():
    # model is already loaded from disk by constructor

    basePath = os.path.join(predictionsBasePath[0])
    if not os.path.exists(basePath):
        os.makedirs(basePath)

    with torch.no_grad():
        for i, data in enumerate(challengeValloader):
            inputs, pids, xOffset, yOffset, zOffset = data
            print('***********************************')
            print("processing {}".format(pids[0]))
            inputs = inputs.to('cuda:0')

            # predict labels and bring into required shape
            outputs = model(inputs)
            outputs = outputs[:, :, :, :, :155]

            # visualize the feature map to tensorboard
            # input_list = [inputs[0:1, 0:1, :, :, 64], inputs[0:1, 1:2, :, :, 64], inputs[0:1, 2:3, :, :, 64],
            #               inputs[0:1, 3:4, :, :, 64]]
            max = inputs.max()
            min = inputs.min()
            print(f'max:{max} min:{min}')
            print('***********************************')
            Flair = [inputs[0:1, 0:1, :, :, 64]]
            T1 = [inputs[0:1, 1:2, :, :, 64]]
            T1ce = [inputs[0:1, 2:3, :, :, 64]]
            T2 = [inputs[0:1, 3:4, :, :, 64]]
            pred_list = [outputs[0:1, :, :, :, 64]]
            board_add_images(writer, f'{pids[0]}/Flair', Flair, 0)
            board_add_images(writer, f'{pids[0]}/T1', T1, 0)
            board_add_images(writer, f'{pids[0]}/T1ce', T1ce, 0)
            board_add_images(writer, f'{pids[0]}/T2', T2, 0)
            board_add_images(writer, f'{pids[0]}/pred', pred_list, 0)

            s = outputs.shape
            fullsize = outputs.new_zeros((s[0], s[1], 240, 240, 155))
            if xOffset + s[2] > 240:
                outputs = outputs[:, :, :240 - xOffset, :, :]
            if yOffset + s[3] > 240:
                outputs = outputs[:, :, :, :240 - yOffset, :]
            if zOffset + s[4] > 155:
                outputs = outputs[:, :, :, :, :155 - zOffset]
            fullsize[:, :, xOffset:xOffset + s[2], yOffset:yOffset + s[3], zOffset:zOffset + s[4]] = outputs

            # binarize output
            wt, tc, et = fullsize.chunk(3, dim=1)
            s = fullsize.shape
            wt = (wt > 0.6).view(s[2], s[3], s[4])
            tc = (tc > 0.5).view(s[2], s[3], s[4])
            et = (et > 0.7).view(s[2], s[3], s[4])

            result = fullsize.new_zeros((s[2], s[3], s[4]), dtype=torch.uint8)
            result[wt] = 2
            result[tc] = 1
            result[et] = 4

            npResult = result.cpu().numpy()
            max = npResult.max()
            min = npResult.min()
            path = os.path.join(basePath, "{}.nii.gz".format(pids[0]))
            utils.save_nii(path, npResult, None, None)

    print("Done :)")


if __name__ == "__main__":
    makePredictions()