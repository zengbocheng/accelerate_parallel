"""
@author: bochengz
@date: 2023/04/14
@email: bochengzeng@bochengz.top
"""

from argparse import ArgumentParser
import yaml
import pprint
import os
import numpy as np
import torch


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--file', dest='filename', required=True)
    return parser


def load_config(yaml_filename):
    if os.path.exists(yaml_filename):
        with open(yaml_filename, 'r', encoding='utf-8') as stream:
            content = yaml.load(stream, Loader=yaml.FullLoader)
        return content
    else:
        print('config file don\'t exist!')
        exit(1)


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def process_config(config):

    # set random seed
    set_random_seed(config['random_seed'])

    # set data type
    if config['precision'] == 'float64':
        config['dtype'] = torch.float64
        torch.set_default_dtype(torch.float64)
    else:
        config['dtype'] = torch.float32
        torch.set_default_dtype(torch.float32)

    # update path in config
    config['log_file'] = config['log_file'].format(config['experiment_name'])
    config['checkpoint_path'] = config['checkpoint_path'].format(config['experiment_name'])
    config['figs_loss_train'] = config['figs_loss_train'].format(config['experiment_name'])

    return config


if __name__ == '__main__':
    from kogger import Logger

    args = get_parser().parse_args()
    config = load_config(yaml_filename=args.filename)
    config = process_config(config)
    logger = Logger('CONFIG')
    logger.info('Load config successfully!')
    logger.info(pprint.pformat(config))
