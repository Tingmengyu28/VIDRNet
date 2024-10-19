import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
# import numpy as np
import d3networks.weight_initialization as w_init

from ipdb import set_trace as st
###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError(f'normalization layer [{norm_type}] is not found')
    return norm_layer


def define_G(input_nc, output_nc=[1], net_architecture='DenseUNet', tasks=['depth'], pretrained=True, n_classes=1000):
    netG = None
    # use_gpu = len(gpu_ids) > 0
    # norm_layer = get_norm_layer(norm_type=norm)

    from .dense_decoders_multitask_auto import denseUnet121
    netG = denseUnet121(pretrained=pretrained,
                        input_nc=input_nc,
                        outputs_nc=output_nc,
                        init_method='normal',
                        use_dropout=True,
                        use_skips=True,
                        d_block_type='basic',
                        num_classes=n_classes,
                        tasks=tasks,
                        type_net=net_architecture)
    # print number of parameters of the network
    print_n_parameters_network(netG)

    if not pretrained:
        w_init.init_weights(netG, 'normal')
    return netG


def print_n_parameters_network(net):
    num_params = sum(param.numel() for param in net.parameters())
    print('Total number of parameters: %d' % num_params)
