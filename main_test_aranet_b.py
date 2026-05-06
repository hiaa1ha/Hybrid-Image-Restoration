import os.path
import logging

import numpy as np
from collections import OrderedDict

import torch

from utils import utils_logger
from utils import utils_model
from utils import utils_image as util


# -------------------------------------------------------
# Inference / evaluation script for ARANet-B.
# Usage:
#   python main_test_aranet_b.py
#
# Edit the configuration block below before running.
# A trained .pth model weight file is required in model_zoo/.
# -------------------------------------------------------


def main():

    # ================================================================
    # Configuration — edit these values before running
    # ================================================================
    noise_level_img   = 25          # noise level added to the test image (sigma)
    noise_level_model = 25          # noise level passed to the model (should match above)
    n_channels        = 3           # 1 = grayscale, 3 = color
    model_name        = 'aranet_b'  # name of the .pth file in model_zoo/ (without extension)
    testset_name      = 'set12'     # folder inside testsets/ to evaluate on
    need_degradation  = True        # True: add synthetic noise; False: images are already degraded
    x8                = False       # True: apply 8-fold self-ensemble (slower, slightly better PSNR)
    show_img          = False       # True: display images during inference (requires a display)
    # ================================================================

    task_current = 'dn'       # denoising task identifier
    sf           = 1          # scale factor (unused for denoising, kept for API compatibility)
    model_pool   = 'model_zoo'
    testsets     = 'testsets'
    results      = 'results'

    result_name  = testset_name + '_' + model_name
    border       = sf if task_current == 'sr' else 0
    model_path   = os.path.join(model_pool, model_name + '.pth')

    # ----------------------------------------
    # Build paths
    # ----------------------------------------
    L_path = os.path.join(testsets, testset_name)   # input (low quality / noisy) images
    H_path = L_path                                  # ground-truth images (same folder when synthesising noise)
    E_path = os.path.join(results, result_name)      # output (estimated / restored) images
    util.mkdir(E_path)

    if H_path == L_path:
        need_degradation = True

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name + '.log'))
    logger = logging.getLogger(logger_name)

    need_H = H_path is not None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # Load ARANet-B model
    # ----------------------------------------
    from models.network_aranet_b import UNetRes as net

    # in_nc = n_channels + 1 because the noise-level map is concatenated to the input
    model = net(in_nc=n_channels + 1, out_nc=n_channels, nf=64, nb=4)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    logger.info('Model path: {:s}'.format(model_path))
    num_params = sum(p.numel() for p in model.parameters())
    logger.info('Parameter count: {:,d}'.format(num_params))

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    logger.info('noise level (image / model): {} / {}'.format(noise_level_img, noise_level_model))
    logger.info('Test images path: {}'.format(L_path))

    L_paths = util.get_image_paths(L_path)
    H_paths = util.get_image_paths(H_path) if need_H else None

    for idx, img in enumerate(L_paths):

        # ----------------------------------------
        # (1) Load and optionally degrade input
        # ----------------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img))
        img_L = util.imread_uint(img, n_channels=n_channels)
        img_L = util.uint2single(img_L)

        if need_degradation:
            np.random.seed(seed=0)   # fixed seed for reproducibility
            img_L = img_L + np.random.normal(0, noise_level_img / 255., img_L.shape)

        util.imshow(util.single2uint(img_L),
                    title='Noisy image  (sigma={})'.format(noise_level_img)) if show_img else None

        # Concatenate the noise-level map as an extra channel
        img_L = util.single2tensor4(img_L)
        noise_map = torch.ones((1, 1, img_L.size(2), img_L.size(3)),
                               dtype=torch.float).mul_(noise_level_model / 255.)
        img_L = torch.cat((img_L, noise_map), dim=1).to(device)

        # ----------------------------------------
        # (2) Run ARANet-B inference
        # ----------------------------------------
        if not x8:
            img_E = model(img_L)
        else:
            # x8 self-ensemble: averages predictions over 8 geometric transforms
            img_E = utils_model.test_mode(model, img_L, mode=3)

        img_E = util.tensor2uint(img_E)

        # ----------------------------------------
        # (3) Compare against ground truth
        # ----------------------------------------
        if need_H:
            img_H = util.imread_uint(H_paths[idx], n_channels=n_channels).squeeze()

            psnr = util.calculate_psnr(img_E, img_H, border=border)
            ssim = util.calculate_ssim(img_E, img_H, border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            logger.info('{:s}  PSNR: {:.2f} dB  SSIM: {:.4f}'.format(img_name + ext, psnr, ssim))

            util.imshow(np.concatenate([img_E, img_H], axis=1),
                        title='Restored / Ground-truth') if show_img else None

        # ----------------------------------------
        # (4) Save restored image
        # ----------------------------------------
        util.imsave(img_E, os.path.join(E_path, img_name + ext))

    # ----------------------------------------
    # Summary
    # ----------------------------------------
    if need_H:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        logger.info('Average PSNR: {:.2f} dB  |  Average SSIM: {:.4f}  '
                    '[{} on {}]'.format(ave_psnr, ave_ssim, model_name, testset_name))


if __name__ == '__main__':
    main()
