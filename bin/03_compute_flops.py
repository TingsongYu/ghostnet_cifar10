# -*- coding: utf-8 -*-
"""
# @file name  : compute_flops.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-02-29
# @brief      : 观察resnet56 及 ghost-resnet-56
"""
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
from torchstat import stat
from models.ghost_net import GhostModule
from models.resnet import resnet56
from models.vgg import VGG
from tools.common_tools import replace_conv


if __name__ == '__main__':

    img_shape = (3, 32, 32)

    resnet56 = resnet56()
    stat(resnet56, img_shape)       # https://github.com/Swall0w/torchstat
    print("↑↑↑↑ is resnet56")
    print("\n"*10)

    ghost_resnet56 = replace_conv(resnet56, GhostModule, arc="resnet56")
    stat(ghost_resnet56, img_shape)
    print("↑↑↑↑ is ghost_resnet56")

    vgg = 0
    # vgg = 1
    if vgg:
        vgg16 = VGG("VGG16")
        stat(vgg16, img_shape)
        print("↑↑↑↑ is vgg16")
        print("\n"*10)

        ghost_vgg16 = replace_conv(vgg16, GhostModule, arc="vgg16")
        stat(ghost_vgg16, img_shape)
        print("↑↑↑↑ is ghost_vgg16")





