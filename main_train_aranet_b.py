import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model


# -------------------------------------------------------
# Training entry point for ARANet-B image restoration.
# Usage:
#   python main_train_aranet_b.py --opt options/train_aranet_b.json
#
# Supports single-GPU and multi-GPU (distributed) training.
# All hyper-parameters are controlled via the JSON option file.
# -------------------------------------------------------


def main(json_path='options/train_aranet_b.json'):

    # ----------------------------------------
    # Step 1: Parse and prepare options
    # ----------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # Distributed training settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    # Create output directories (logs, models, images)
    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # Resume from checkpoint if available
    # ----------------------------------------
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    opt['path']['pretrained_netG'] = init_path_G
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_optimizerG)

    border = opt['scale']

    # Save the resolved option dict alongside the experiment
    if opt['rank'] == 0:
        option.save(opt)

    # Convert missing keys to None instead of raising KeyError
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # Configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # Reproducibility seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ----------------------------------------
    # Step 2: Build dataloaders
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set,
                                                   shuffle=dataset_opt['dataloader_shuffle'],
                                                   drop_last=True,
                                                   seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'] // opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers'] // opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    # ----------------------------------------
    # Step 3: Initialise ARANet-B model
    # ----------------------------------------
    model = define_Model(opt)
    model.init_train()
    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    # ----------------------------------------
    # Step 4: Main training loop
    # ----------------------------------------
    for epoch in range(1000000):  # runs until manually stopped
        if opt['dist']:
            train_sampler.set_epoch(epoch)

        for i, train_data in enumerate(train_loader):

            current_step += 1

            # 1) Update learning rate schedule
            model.update_learning_rate(current_step)

            # 2) Feed a batch of noisy/clean patch pairs
            model.feed_data(train_data)

            # 3) Forward pass + loss + backward
            model.optimize_parameters(current_step)

            # 4) Print training log
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # 5) Save model checkpoint
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # 6) Run evaluation on test set and log PSNR
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:

                avg_psnr = 0.0
                idx = 0

                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])

                    # Save the restored image
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                    util.imsave(E_img, save_img_path)

                    # Compute PSNR against ground truth
                    current_psnr = util.calculate_psnr(E_img, H_img, border=border)
                    logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))
                    avg_psnr += current_psnr

                avg_psnr = avg_psnr / idx
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n'.format(
                    epoch, current_step, avg_psnr))


if __name__ == '__main__':
    main()
