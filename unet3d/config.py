import argparse

import os
import torch
import yaml

# DEFAULT_DEVICE = 'cuda:1'


def load_config():
    parser = argparse.ArgumentParser(description='UNet3D training')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    args = parser.parse_args()
    config = _load_config_yaml(args.config)
    # Get a device to train on
    # os.environ['CUDA_VISIBLE_DEVICES'] = config['default_device']
    # device = config.get('device', config['default_device'])
    # config['device'] = torch.device(device)
    return config


def _load_config_yaml(config_file):
    return yaml.load(open(config_file, 'r'))
