"""
select_network.py

Network factory for ARANet-B. Routes the net_type key from the option
JSON to the ARANet-B architecture class and initialises its weights.
"""

import functools
import torch
from torch.nn import init


# ----------------------------------------
# Generator — ARANet-B
# ----------------------------------------
def define_G(opt):
    opt_net = opt['netG']
    net_type = opt_net['net_type']

    if net_type == 'aranet_b':
        from models.network_aranet_b import UNetRes as net
        netG = net(
            in_nc=opt_net['in_nc'],
            out_nc=opt_net['out_nc'],
            nf=opt_net['nc'][0] if isinstance(opt_net['nc'], list) else opt_net['nc'],
            nb=opt_net['nb']
        )
    else:
        raise NotImplementedError('netG [{:s}] is not found.'.format(net_type))

    # ----------------------------------------
    # Initialise weights
    # ----------------------------------------
    if opt['is_train']:
        init_weights(
            netG,
            init_type=opt_net['init_type'],
            init_bn_type=opt_net['init_bn_type'],
            gain=opt_net['init_gain']
        )

    return netG


# ----------------------------------------
# Weight initialisation utility
# ----------------------------------------
def init_weights(net, init_type='orthogonal', init_bn_type='uniform', gain=0.2):
    """
    Supported init_type values:
        normal, uniform, xavier_normal, xavier_uniform,
        kaiming_normal, kaiming_uniform, orthogonal, default/none
    Supported init_bn_type values:
        uniform, constant
    """

    def init_fn(m, init_type='orthogonal', init_bn_type='uniform', gain=0.2):
        classname = m.__class__.__name__

        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            if init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)
            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)
            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)
            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)
            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('Initialisation method [{:s}] is not implemented'.format(init_type))

            if m.bias is not None:
                m.bias.data.zero_()

        elif classname.find('BatchNorm2d') != -1:
            if init_bn_type == 'uniform':
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            elif init_bn_type == 'constant':
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError('Initialisation method [{:s}] is not implemented'.format(init_bn_type))

    if init_type not in ['default', 'none']:
        print('Weight init: [{:s} + {:s}], gain = {:.2f}'.format(init_type, init_bn_type, gain))
        fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
        net.apply(fn)
    else:
        print('Skipping weight init — handled inside network definition.')
