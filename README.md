[![DOI](https://zenodo.org/badge/149826542.svg)](https://zenodo.org/badge/latestdoi/149826542)

# label-mixup

Our baseline model is [NNUnet](https://arxiv.org/abs/1809.10483) and the implementation is based on:

https://github.com/RobinBruegger/PartiallyReversibleUnet

## Prerequisites
- Linux
- NVIDIA GPU
- CUDA CuDNN

## Getting Started

### Dependencies
- pytorch (0.4.1+)
- torchvision (0.2.1+)
- tensorboardx (1.6+)
- h5py
- scipy 
- scikit-image
- pytest
- nibabel

Setup a new conda environment with the required dependencies via:
```
conda create -n 3dunet pytorch torchvision tensorboardx h5py scipy scikit-image pyyaml pytest -c conda-forge -c pytorch
```
Activate newly created conda environment via:
```
source activate 3dunet
```

## Supported model architectures
- in order to train standard 3D U-Net specify `name: UNet3D` in the `model` section of the [config file](resources/train_config_ce.yaml)
- in order to train Residual U-Net specify `name: ResidualUNet3D` in the `model` section of the [config file](resources/train_config_ce.yaml)

## Supported Loss Functions
For a detailed explanation of the loss functions used see:
[Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations](https://arxiv.org/pdf/1707.03237.pdf)
Carole H. Sudre, Wenqi Li, Tom Vercauteren, Sebastien Ourselin, M. Jorge Cardoso

- _WeightedCrossEntropyLoss_ (see 'Weighted cross-entropy (WCE)' in the above paper for a detailed explanation; one can specify class weights via `weight: [w_1, ..., w_k]` in the `loss` section of the config)
- _CrossEntropyLoss_ (one can specify class weights via `weight: [w_1, ..., w_k]` in the `loss` section of the config)
- _PixelWiseCrossEntropyLoss_ (one can specify not only class weights but also per pixel weights in order to give more/less gradient in some regions of the ground truth)
- _BCEWithLogitsLoss_
- _DiceLoss_ standard Dice loss (see 'Dice Loss' in the above paper for a detailed explanation).
- _GeneralizedDiceLoss_ (see 'Generalized Dice Loss (GDL)' in the above paper for a detailed explanation; one can specify class weights via `weight: [w_1, ..., w_k]` in the `loss` section of the config). 
Note: use this loss function only if the labels in the training dataset are very imbalanced
e.g. one class having at lease 3 orders of magnitude more voxels than the others. Otherwise use standard _DiceLoss_ which works better than GDL most of the time. 


## Supported Evaluation Metrics
- **MeanIoU** - Mean intersection over union
- **DiceCoefficient** - Dice Coefficient (computes per channel Dice Coefficient and returns the average)
- **BoundaryAveragePrecision** - Average Precision (normally used for evaluating instance segmentation, however it can be used when the 3D UNet is used to predict the boundary signal from the instance segmentation ground truth)
- **AdaptedRandError** - Adapted Rand Error (see http://brainiac2.mit.edu/SNEMI3D/evaluation for a detailed explanation)

If not specified `MeanIoU` will be used by default.

## Proprecess

1. Install [ANTs N4BiasFieldCorrection](https://github.com/stnava/ANTs/releases) and add the location of the ANTs binaries to the PATH environmental variable.
2. Add the repository directory to the `PYTONPATH` system variable:

```
$ export PYTHONPATH=${PWD}:$PYTHONPATH
```

3. Import the conversion function and run the preprocessing:

```
$ python
>>> from preprocess/preprocess.py import convert_brats_data
>>> convert_brats_data("data/original", "data/preprocessed")
```

4. Generate HDF5 training datasets:

```
python datasets/brats_data_loader.py
```

5. Generate HDF5 validation datasets:

```
python datasets/brats18_validation_data_loader.py
```

## Train
```yaml
python train.py --config resources/train_config_2.yaml 
```

See the [train_config_2.yaml](resources/train_config_2.yaml) for more info.

In order to train on your own data just provide the paths to your HDF5 training and validation datasets in the [train_config_2.yaml](resources/train_config_2.yaml).
The HDF5 files should contain the raw/label data sets in the following axis order: `DHW` (in case of 3D) `CDHW` (in case of 4D).

Monitor progress with Tensorboard `tensorboard --logdir ./3dunet/logs/ --port 8666` (you need `tensorflow` installed in your conda env).
![3dunet-training](https://user-images.githubusercontent.com/706781/45916217-9626d580-be62-11e8-95c3-508e2719c915.png)

### Training tips
1. In order to train with `BCEWithLogitsLoss`, `DiceLoss` or `GeneralizedDiceLoss` the label data has to be 4D (one target binary mask per channel).
If you have a 3D binary data (foreground/background), you can just change `ToTensor` transform for the label to contain `expand_dims: true`, see e.g. [train_config_dice.yaml](resources/train_config_dice.yaml).

2. When training with binary-based losses (`BCEWithLogitsLoss`, `DiceLoss`, `GeneralizedDiceLoss`) `final_sigmoid=True` has to be present in the training config, since every output channel gives the probability of the foreground.
When training with cross entropy based losses (`WeightedCrossEntropyLoss`, `CrossEntropyLoss`, `PixelWiseCrossEntropyLoss`) set `final_sigmoid=False` so that `Softmax` normalization is applied to the output.

## Test
Test on randomly generated 3D volume (just for demonstration purposes) from [random_label3D.h5](resources/random_label3D.h5). 
```
python test.py --config resources/train_config_2.yaml
```
Prediction masks will be saved to `pred_path` defined in the `train_config_.yaml`

### Prediction tips
In order to avoid block artifacts in the output prediction masks the patch predictions are averaged, so make sure that `patch/stride` params lead to overlapping blocks, e.g. `patch: [64 128 128] stride: [32 96 96]` will give you a 'halo' of 32 voxels in each direction.

## Contribute
If you want to contribute back, please make a pull request.

## Cite
If you use this code for your research, please cite as:

Adrian Wolny. (2019, May 7). wolny/pytorch-3dunet: PyTorch implementation of 3D U-Net (Version v1.0.0). Zenodo. http://doi.org/10.5281/zenodo.2671581



