import logging
import os

import numpy as np
import torch
import datetime
from unet3d.config import load_config
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from preprocess.partitioning import get_all_partition_ids
from numpy import random
from visualization import board_add_images
import BraTS

from . import utils


class UNet3DTrainer:
    """3D UNet trainer.

    Args:
        model (Unet3D): UNet 3D model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
            WARN: bear in mind that lr_scheduler.step() is invoked after every validation step
            (i.e. validate_after_iters) not after every epoch. So e.g. if one uses StepLR with step_size=30
            the learning rate will be adjusted after every 30 * validate_after_iters iterations.
        loss_criterion (callable): loss function
        eval_criterion (callable): used to compute training/validation metric (such as Dice, IoU, AP or Rand score)
            saving the best checkpoint is based on the result of this function on the validation set
        device (torch.device): device to train on
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        eval_score_higher_is_better (bool): if True higher eval scores are considered better
        best_eval_score (float): best validation score so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
    """

    def __init__(self, model, optimizer, lr_scheduler, loss_criterion,
                 eval_criterion, device, loaders, checkpoint_dir, model_name,
                 max_num_epochs=100, max_num_iterations=1e5,
                 validate_after_iters=100, log_after_iters=100,
                 validate_iters=None, num_iterations=1, num_epoch=0,
                 eval_score_higher_is_better=True, best_eval_score=None,
                 logger=None):
        if logger is None:
            self.logger = utils.get_logger('VaeUnetTrainer', level=logging.DEBUG)
        else:
            self.logger = logger
        self.config = load_config()
        self.logger.info(model)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.device = device
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.eval_score_higher_is_better = eval_score_higher_is_better
        self.logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')

        if best_eval_score is not None:
            self.best_eval_score = best_eval_score
        else:
            # initialize the best_eval_score
            if eval_score_higher_is_better:
                self.best_eval_score = float('-inf')
            else:
                self.best_eval_score = float('+inf')

        self.writer = SummaryWriter(logdir=os.path.join(checkpoint_dir, self._get_job_name()))

        self.num_iterations = num_iterations
        self.num_epoch = num_epoch

    def _get_job_name(self):
        now = '{:%Y-%m-%d.%H:%M}'.format(datetime.datetime.now())
        return "%s_model_%s" % (now, self.model_name)

    @classmethod
    def from_checkpoint(cls, checkpoint_path, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders,
                        logger=None):
        logger.info(f"Loading checkpoint '{checkpoint_path}'...")
        state = utils.load_checkpoint(checkpoint_path, model, optimizer)
        logger.info(
            f"Checkpoint loaded. Epoch: {state['epoch']}. Best val score: {state['best_eval_score']}. Num_iterations: {state['num_iterations']}")
        checkpoint_dir = os.path.split(checkpoint_path)[0]
        return cls(model, optimizer, lr_scheduler,
                   loss_criterion, eval_criterion,
                   torch.device(state['device']),
                   loaders, checkpoint_dir,
                   eval_score_higher_is_better=state['eval_score_higher_is_better'],
                   best_eval_score=state['best_eval_score'],
                   num_iterations=state['num_iterations'],
                   num_epoch=state['epoch'],
                   max_num_epochs=state['max_num_epochs'],
                   max_num_iterations=state['max_num_iterations'],
                   validate_after_iters=state['validate_after_iters'],
                   log_after_iters=state['log_after_iters'],
                   validate_iters=state['validate_iters'],
                   model_name="VaeUNet",
                   logger=logger)

    @classmethod
    def from_pretrained(cls, pre_trained, model, optimizer, lr_scheduler, loss_criterion, eval_criterion,
                        device, loaders,
                        max_num_epochs=100, max_num_iterations=1e5,
                        validate_after_iters=100, log_after_iters=100,
                        validate_iters=None, num_iterations=1, num_epoch=0,
                        eval_score_higher_is_better=True, best_eval_score=None,
                        logger=None):
        logger.info(f"Logging pre-trained model from '{pre_trained}'...")
        utils.load_checkpoint(pre_trained, model, None)
        checkpoint_dir = os.path.split(pre_trained)[0]
        return cls(model, optimizer, lr_scheduler,
                   loss_criterion, eval_criterion,
                   device, loaders, checkpoint_dir,
                   eval_score_higher_is_better=eval_score_higher_is_better,
                   best_eval_score=best_eval_score,
                   num_iterations=num_iterations,
                   num_epoch=num_epoch,
                   max_num_epochs=max_num_epochs,
                   max_num_iterations=max_num_iterations,
                   validate_after_iters=validate_after_iters,
                   log_after_iters=log_after_iters,
                   validate_iters=validate_iters,
                   logger=logger)

    def fit(self):
        for _ in range(self.num_epoch, self.max_num_epochs):
            # train for one epoch
            should_terminate = self.train(self.loaders['train'])

            if should_terminate:
                break
            if self.config['optimizer']['mode'] == 'SWA':
                self.optimizer.swap_swa_sgd()
            self.num_epoch += 1

    def draw_picture(self, sample):

        fig = plt.figure()
        feature_image = fig.add_subplot(1, 1, 1)
        plt.imshow(sample, cmap="gray")
        feature_image.set_title('output')
        plt.savefig('picture/{}.png'.format(random.randint(1, 1000)))
        plt.close()

    def train(self, train_loader, is_choose_randomly=False, is_mixup=False):
        """Trains the model for 1 epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): training data loader

        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        def _make_crop(input):
            image = input[..., 40:200, 40:200, 21:117]
            # image = input[..., 40:200, 24:216, 13:141]
            return image

        def _make_one_hot(input):
            seg = np.eye(5)[input].transpose(3, 0, 1, 2)
            seg = seg[[0, 1, 2, 4], :, :, :]
            return seg

        def _preprocessing_images(image):
            _image = _make_crop(image)
            _image_max = np.max(_image)
            import augment.transforms as transforms
            transforms = transforms.Compose(
                [transforms.ToTensor(expand_dims=True),
                 transforms.RangeNormalize(max_value=_image_max),
                 transforms.Normalize(std=0.5, mean=0.5)])
            _image = transforms(_image).unsqueeze(0)
            _image = _image.to(self.device)
            return _image

        def _preprocessing_labels(label):
            _label = _make_crop(label)
            _label = _make_one_hot(_label)
            _label = torch.from_numpy(_label).unsqueeze(0)
            _label = _label.to(self.device)
            return _label

        train_losses = utils.RunningAverage()
        train_eval_scores_multi = utils.RunningAverageMulti()

        # sets the model in training mode
        self.model.train()

        if is_choose_randomly:
            train_ids, test_ids, validation_ids = get_all_partition_ids()
            train_id_list = []
            for train_id in train_ids:
                train_id_list.append(train_id)

            loaders_config = self.config['loaders']
            data_paths = loaders_config['dataset_path']
            brats = BraTS.DataSet(brats_root=data_paths[0], year=2019)
            index = np.random.permutation(len(train_id_list))

            for i in range(0, len(train_id_list)):

                if is_mixup:
                    idx = random.randint(0, len(train_id_list))
                    patient1 = brats.train.patient(train_id_list[idx])
                    patient2 = brats.train.patient(train_id_list[index[idx]])
                    images1 = _preprocessing_images(patient1.mri)
                    images2 = _preprocessing_images(patient2.mri)
                    labels1 = _preprocessing_labels(patient1.seg)
                    labels2 = _preprocessing_labels(patient2.seg)
                    alpha = 1.0  # 超参数
                    lam = np.random.beta(alpha, alpha)
                    input = lam * images1 + (1 - lam) * images2
                    target = lam * labels1 + (1 - lam) * labels2
                    # print("eq sum target", target.eq(labels1.data).cpu().sum())
                    self.logger.info(
                        f'Using mixup. Training iteration {self.num_iterations}. '
                        f'Batch {i}. Epoch [{self.num_epoch}/{self.max_num_epochs - 1}]')
                else:
                    idx = random.randint(0, len(train_id_list))
                    patient = brats.train.patient(train_id_list[idx])
                    images = patient.mri
                    labels = patient.seg
                    input = _preprocessing_images(images)
                    target = _preprocessing_labels(labels)
                    self.logger.info(f'Patient ID {train_id_list[idx]}. Training iteration {self.num_iterations}. '
                                     f'Batch {i}. Epoch [{self.num_epoch}/{self.max_num_epochs - 1}]')

                output, loss = self._forward_pass(input, target)

                train_losses.update(loss.item())

                # compute gradients and update parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.num_iterations % self.validate_after_iters == 0:
                    # evaluate on validation set
                    eval_score = self.validate(self.loaders['val'])
                    # log current learning rate in tensorboard
                    self._log_lr()
                    # remember best validation metric
                    is_best = self._is_best_eval_score(eval_score)

                    # save checkpoint
                    self._save_checkpoint(is_best)

                if self.num_iterations % self.log_after_iters == 0:
                    # if model contains final_activation layer for normalizing logits apply it, otherwise both
                    # the evaluation metric as well as images in tensorboard will be incorrectly computed
                    if hasattr(self.model, 'final_activation'):
                        output = self.model.final_activation(output)

                    # visualize the feature map to tensorboard
                    board_list = [input[0:1, 1:4, :, :, 64], output[0:1, 1:4, :, :, 64], target[0:1, 1:4, :, :, 64]]
                    board_add_images(self.writer, 'feature map', board_list, self.num_iterations)

                    # compute eval criterion
                    eval_score = self.eval_criterion(output, target)
                    # train_eval_scores.update(eval_score.item(), self._batch_size(input))
                    train_eval_scores_multi.update(eval_score, self._batch_size(input))

                    # log stats, params and images
                    self.logger.info(
                        f'Training stats. Loss: {train_losses.avg}. '
                        f'Evaluation score WT:{train_eval_scores_multi.avg1}, '
                        f'TC:{train_eval_scores_multi.avg2}, '
                        f'ET:{train_eval_scores_multi.avg3}')
                    self._log_stats_multi('train', train_losses.avg, train_eval_scores_multi.avg1,
                                          train_eval_scores_multi.avg2, train_eval_scores_multi.avg3)
                    self._log_params()

                    # self._log_images(input, target, output)

                if self.max_num_iterations < self.num_iterations:
                    self.logger.info(
                        f'Maximum number of iterations {self.max_num_iterations} exceeded. Finishing training...')
                    return True

                self.num_iterations += 1
        else:

            for i, t in enumerate(train_loader):

                self.logger.info(
                    f'Training iteration {self.num_iterations}. Batch {i}. Epoch [{self.num_epoch}/{self.max_num_epochs - 1}]')

                input, target, weight = self._split_training_batch(t)

                output, loss = self._forward_pass(input, target, weight)

                # output_sample = output[0, 1, :, :, 80].cpu().detach().numpy()
                # self.draw_picture(output_sample)

                train_losses.update(loss.item(), self._batch_size(input))

                # compute gradients and update parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.num_iterations % self.validate_after_iters == 0:
                    # evaluate on validation set
                    eval_score = self.validate(self.loaders['val'])
                    # adjust learning rate if necessary
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        # self.scheduler.step(eval_score)
                        pass
                    else:
                        # self.scheduler.step()
                        pass
                    # log current learning rate in tensorboard
                    self._log_lr()
                    # remember best validation metric
                    is_best = self._is_best_eval_score(eval_score)

                    # save checkpoint
                    self._save_checkpoint(is_best)

                if self.num_iterations % self.log_after_iters == 0:
                    # if model contains final_activation layer for normalizing logits apply it, otherwise both
                    # the evaluation metric as well as images in tensorboard will be incorrectly computed
                    if hasattr(self.model, 'final_activation'):
                        output = self.model.final_activation(output)

                    # visualize the feature map to tensorboard
                    board_list = [input[0:1, 1:4, :, :, 64], output[0:1, :, :, :, 64], target[0:1, :, :, :, 64]]
                    board_add_images(self.writer, 'feature map', board_list, self.num_iterations)

                    # compute eval criterion
                    eval_score = self.eval_criterion(output, target)
                    # train_eval_scores.update(eval_score.item(), self._batch_size(input))
                    train_eval_scores_multi.update(eval_score, self._batch_size(input))

                    # log stats, params and images
                    self.logger.info(
                        f'Training stats. Loss: {train_losses.avg}. '
                        f'Evaluation score WT:{train_eval_scores_multi.avg1}, '
                        f'TC:{train_eval_scores_multi.avg2}, '
                        f'ET:{train_eval_scores_multi.avg3}')
                    self._log_stats_multi('train', train_losses.avg, train_eval_scores_multi.avg1,
                                          train_eval_scores_multi.avg2, train_eval_scores_multi.avg3)
                    self._log_params()

                    # self._log_images(input, target, output)

                if self.max_num_iterations < self.num_iterations:
                    self.logger.info(
                        f'Maximum number of iterations {self.max_num_iterations} exceeded. Finishing training...')
                    return True

                self.num_iterations += 1

        return False

    def validate(self, val_loader):
        self.logger.info('Validating...')

        val_losses = utils.RunningAverage()
        # val_scores = utils.RunningAverage()
        val_scores_multi = utils.RunningAverageMulti()

        try:
            # set the model in evaluation mode; final_activation doesn't need to be called explicitly
            self.model.eval()
            with torch.no_grad():
                for i, t in enumerate(val_loader):
                    self.logger.info(f'Validation iteration {i}')

                    input, target, weight = self._split_training_batch(t)

                    output, loss = self._forward_pass(input, target, weight)
                    val_losses.update(loss.item(), self._batch_size(input))

                    eval_score = self.eval_criterion(output, target)

                    # val_scores.update(eval_score.item(), self._batch_size(input))
                    val_scores_multi.update(eval_score, self._batch_size(input))

                    if self.validate_iters is not None and self.validate_iters <= i:
                        # stop validation
                        break

                # self._log_stats('val', val_losses.avg, val_scores.avg)
                self._log_stats_multi('val', val_losses.avg, val_scores_multi.avg1, val_scores_multi.avg2, val_scores_multi.avg3)
                # self.logger.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
                self.logger.info(f'Validation finished. Loss: {val_losses.avg}. '
                                 f'Evaluation score WT:{val_scores_multi.avg1}, '
                                 f'TC:{val_scores_multi.avg2}, '
                                 f'ET:{val_scores_multi.avg3}')

                return val_scores_multi.avg1
        finally:
            # set back in training mode
            self.model.train()

    def _split_training_batch(self, t):
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_device(x) for x in input])
            else:
                return input.to(self.device, dtype=torch.float)

        t = _move_to_device(t)
        weight = None
        if len(t) == 2:
            input, target = t
        else:
            input, target, weight = t
        return input, target, weight

    def _forward_pass(self, input, target, weight=None):
        # forward pass
        output = self.model(input)

        # compute the loss
        if self.model_name == "VaeUNet":
            loss = self.loss_criterion(input, output[1], output[0], target, output[2], output[3])
            output = output[0]
        else:
            if weight is None:
                loss = self.loss_criterion(output, target)
            else:
                loss = self.loss_criterion(output, target, weight)

        return output, loss

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            self.logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    def _save_checkpoint(self, is_best):
        utils.save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': self.model.state_dict(),
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
            'max_num_iterations': self.max_num_iterations,
            'validate_after_iters': self.validate_after_iters,
            'log_after_iters': self.log_after_iters,
            'validate_iters': self.validate_iters
        }, is_best, checkpoint_dir=self.checkpoint_dir,
            logger=self.logger)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_stats(self, phase, loss_avg, eval_score_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_eval_score_avg': eval_score_avg
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_stats_multi(self, phase, loss_avg, eval_score_avg1, eval_score_avg2, eval_score_avg3):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_eval_score_avg1': eval_score_avg1,
            f'{phase}_eval_score_avg2': eval_score_avg2,
            f'{phase}_eval_score_avg3': eval_score_avg3
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)


    def _log_params(self):
        self.logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(self, input, target, prediction):
        inputs_map = {
            'inputs': input,
            'targets': target,
            'predictions': prediction
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self._images_from_batch(name, batch):
                self.writer.add_image(tag, image, self.num_iterations, dataformats='HW')

    def _images_from_batch(self, name, batch):
        tag_template = '{}/batch_{}/channel_{}/slice_{}'

        tagged_images = []

        if batch.ndim == 5:
            # NCHWD
            slice_idx = batch.shape[4] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                for channel_idx in range(batch.shape[1]):
                    tag = tag_template.format(name, batch_idx, channel_idx, slice_idx)
                    img = batch[batch_idx, channel_idx, slice_idx, ...]
                    tagged_images.append((tag, self._normalize_img(img)))
        else:
            # batch has no channel dim: NHWD
            slice_idx = batch.shape[3] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                tag = tag_template.format(name, batch_idx, 0, slice_idx)
                img = batch[batch_idx, slice_idx, ...]
                tagged_images.append((tag, self._normalize_img(img)))

        return tagged_images

    @staticmethod
    def _normalize_img(img):
        return (img - np.min(img)) / np.ptp(img)

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)
