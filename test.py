from unet3d.config import load_config
import os
import torch
import torch.nn as nn
from datasets.hdf5 import BratsDataset
# from unet3d.model import get_model
from unet3d import utils
from tqdm import tqdm
import numpy as np
from datasets.hdf5 import get_brats_train_loaders

logger = utils.get_logger('UNet3DPredictor')

# Load and log experiment configuration
config = load_config()

# Load model state
# model = get_model(config)
# model_path = config['trainer']['test_model']
# logger.info(f'Loading model from {model_path}...')
# utils.load_checkpoint(model_path, model)

# Run on GPU or CPU
# if torch.cuda.is_available():
#     print("using cuda (", torch.cuda.device_count(), "device(s))")
#     if torch.cuda.device_count() > 1:
#         model = nn.DataParallel(model)
#     device = torch.device("cuda:1")
# else:
#     device = torch.device("cpu")
#     print("using cpu")
# model = model.to(device)
logger.info(f"Sending the model to '{config['device']}'")
# model1 = torch.load('/home/dell/data/Dataset/Brats19/pytorch-3dunet/checkpoints/muticlass_GN_NNNetbaseline_LookAhead_batchsize=1_fold3/epoch189_model.pkl').to(config['device'])
# model2 = torch.load('/home/dell/data/Dataset/Brats19/pytorch-3dunet/checkpoints/N4bias_muticlass_labelmix_teacher*2_GN_NNNetbaseline_LookAhead_batchsize=1_fold3/epoch73_model.pkl').to(config['device'])
# model3 = torch.load('/home/dell/data/Dataset/Brats19/pytorch-3dunet/checkpoints/N4bias_muticlass_labelmix_teacher*2_GN_NNNetbaseline_LookAhead_batchsize=1_fold3/epoch141_model.pkl').to(config['device'])
# model4 = torch.load('/home/dell/data/Dataset/Brats19/pytorch-3dunet/checkpoints/N4bias_muticlass_labelmix_teacher*2_GN_NNNetbaseline_LookAhead_batchsize=1_fold3/epoch114_model.pkl').to(config['device'])
# model5 = torch.load('/home/dell/data/Dataset/Brats19/pytorch-3dunet/checkpoints/N4bias_muticlass_labelmix_teacher*2_GN_NNNetbaseline_LookAhead_batchsize=1_fold2/epoch120_model.pkl').to(config['device'])
# model6 = torch.load('/home/dell/data/Dataset/Brats19/pytorch-3dunet/checkpoints/muticlass_N4bias__GN_NNNetbaseline_ET_LookAhead_batchsize=1/epoch183_model.pkl').to(config['device'])
# model7 = torch.load('/home/dell/data/Dataset/Brats19/pytorch-3dunet/checkpoints/muticlass_N4bias__GN_NNNetbaseline_ET_LookAhead_batchsize=1/epoch177_model.pkl').to(config['device'])
# model8 = torch.load('/home/dell/data/Dataset/Brats19/pytorch-3dunet/checkpoints/muticlass_N4bias__GN_NNNetbaseline_ET_LookAhead_batchsize=1/epoch100_model.pkl').to(config['device'])
# model9 = torch.load('/home/dell/data/Dataset/Brats19/pytorch-3dunet/checkpoints/test/epoch1_model.pkl').to(config['device'])
# model10 = torch.load('/home/dell/data/Dataset/Brats19/pytorch-3dunet/checkpoints/test/epoch29_model.pkl').to(config['device'])
# model11 = torch.load('/home/dell/data/Dataset/Brats19/pytorch-3dunet/checkpoints/muticlass_GN_NNNetbaseline_LookAhead_batchsize=1_fold1/epoch111_model.pkl').to(config['device'])

# model1 = torch.load('/home/dell/data/Dataset/Brats19/pytorch-3dunet/checkpoints/muticlass_mixlabel_teacher*2_GN_NNNetbaseline_LookAhead_batchsize=1_fold0/epoch196_model.pkl').to(config['device'])
# model2 = torch.load('/home/dell/data/Dataset/Brats19/pytorch-3dunet/checkpoints/muticlass_mixlabel_teacher*2_GN_NNNetbaseline_LookAhead_batchsize=1_fold0/epoch180_model.pkl').to(config['device'])
model3 = torch.load('/home/dell/data/Dataset/Brats19/pytorch-3dunet/checkpoints/muticlass_mixlabel_teacher*2_GN_NNNetbaseline_LookAhead_batchsize=1_fold0/epoch146_model.pkl', map_location="cpu").to(config['device'])
# model4 = torch.load( '/home/dell/data/Dataset/Brats19/pytorch-3dunet/checkpoints/muticlass_GN_NNNetbaseline_LookAhead_batchsize=1_fold4/epoch153_model.pkl').to(config['device'])
model5 = torch.load('/home/dell/data/Dataset/Brats19/pytorch-3dunet/checkpoints/muticlass_GN_NNNetbaseline_LookAhead_batchsize=1_fold4/epoch166_model.pkl').to(config['device'])
# model16 = torch.load('/home/dell/data/Dataset/Brats19/pytorch-3dunet/checkpoints/muticlass_newbias_teachernetwork*3_GN_NNNetbaseline_LookAhead_batchsize=1_fold0/epoch75_model.pkl').to(config['device'])
model17 = torch.load('/home/dell/data/Dataset/Brats19/pytorch-3dunet/checkpoints/muticlass_newbias_teachernetwork*3_GN_NNNetbaseline_LookAhead_batchsize=1_fold0/epoch57_model.pkl', map_location="cpu").to(config['device'])
model18 = torch.load('/home/dell/data/Dataset/Brats19/pytorch-3dunet/checkpoints/muticlass_newbias_teachernetwork*3_GN_NNNetbaseline_LookAhead_batchsize=1_fold0/epoch83_model.pkl', map_location="cpu").to(config['device'])
predictionsBasePath = config['loaders']['pred_path']
BRATS_VAL_PATH = config['loaders']['test_path']

loaders = get_brats_train_loaders(config)
challenge_loader = loaders['challenge']


def makePredictions(challenge_loader):
    # model is already loaded from disk by constructor
    basePath = os.path.join(predictionsBasePath[0] + f"/model3,5,17,18")
    if not os.path.exists(basePath):
        os.makedirs(basePath)
    with torch.no_grad():
        for i, data in enumerate(tqdm(challenge_loader)):
            inputs, pids, xOffset, yOffset, zOffset = data
            print("processing {}".format(pids[0]))
            inputs = inputs.to(config['device'])

            output_ensemble = 0
            for i, model in enumerate([model18, model3, model5, model17]):
                model.eval()
                # predict labels and bring into required shape
                outputs, _ = model(inputs)
                # TTA
                outputs += model(inputs.flip(dims=(2,)))[0].flip(dims=(2,))
                outputs += model(inputs.flip(dims=(3,)))[0].flip(dims=(3,))
                outputs += model(inputs.flip(dims=(4,)))[0].flip(dims=(4,))
                outputs += model(inputs.flip(dims=(2, 3)))[0].flip(dims=(2, 3))
                outputs += model(inputs.flip(dims=(2, 4)))[0].flip(dims=(2, 4))
                outputs += model(inputs.flip(dims=(3, 4)))[0].flip(dims=(3, 4))
                outputs += model(inputs.flip(dims=(2, 3, 4)))[0].flip(dims=(2, 3, 4))
                outputs = outputs / 8.0  # mean
                # model ensemble
                output_ensemble += outputs
            outputs = output_ensemble[:, :, :, :, :155] / 4.0
            print(f"pred max is {torch.max(outputs)}")
            print(f"pred min is {torch.min(outputs)}")
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
            wt = (wt > 0.5).view(s[2], s[3], s[4])
            tc = (tc > 0.5).view(s[2], s[3], s[4])
            et = (et > 0.5).view(s[2], s[3], s[4])

            result = fullsize.new_zeros((s[2], s[3], s[4]), dtype=torch.uint8)
            result[wt] = 2
            result[tc] = 1
            result[et] = 4

            npResult = result.cpu().numpy()
            ET_voxels = (npResult == 4).sum()
            if ET_voxels < 100:
                # torch.where(result == 4, result, torch.ones_like(result))
                npResult[np.where(npResult == 4)] = 1

            path = os.path.join(basePath, "{}.nii.gz".format(pids[0]))
            utils.save_nii(path, npResult, None, None)

    print("Done :)")


if __name__ == "__main__":
    makePredictions(challenge_loader)