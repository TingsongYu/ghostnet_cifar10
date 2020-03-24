# -*- coding: utf-8 -*-
"""
# @file name  : fmap_visulization.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-02-29
# @brief      : 特征图可视化
"""
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
import torchvision.utils as vutils
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tools.common_tools import *
from models.ghost_net import GhostModule
from models.resnet import resnet56
from models.vgg import VGG
from config.config import class_names, norm_mean, norm_std


def farward_hook(module, input, output):
    fmap_block.append(output)


if __name__ == '__main__':

    path_img = os.path.join(BASE_DIR, "..", "data", "cifar10_train", "5", "5_11519.png")
    path_checkpoint = os.path.join(BASE_DIR, "..", "results", "ghost-vgg-16", "checkpoint_best.pkl")

    time_str = datetime.strftime(datetime.now(), '%m-%d_%H-%M')
    log_dir = os.path.join(BASE_DIR, "..", "results", "runs", time_str)
    writer = SummaryWriter(log_dir=log_dir, comment='none', filename_suffix="none")

    # load img
    img_input = img_preprocess(path_img, norm_mean, norm_std)

    # load model
    # model = resnet56()
    # model = replace_conv(model, GhostModule, "resnet56")

    model = VGG("VGG16")
    model = replace_conv(model, GhostModule, "vgg16")   # ghost-vgg16

    check_p = torch.load(path_checkpoint, map_location="cpu", encoding='iso-8859-1')
    pretrain_dict = check_p["model_state_dict"]
    state_dict_cpu = state_dict_to_cpu(pretrain_dict)
    model.load_state_dict(state_dict_cpu)

    # register hook
    fmap_block, fmap_names = [], []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(farward_hook)
            fmap_names.append(name)

    # forward
    output = model(img_input)
    idx = np.argmax(output.cpu().data.numpy())
    print("predict: {}".format(class_names[idx]))

    # ----------------------------------- feature maps visualization -----------------------------------
    for (fmap, f_name) in zip(fmap_block, fmap_names):
        b, c, h, w = fmap.shape
        nrow = int(np.sqrt(c))
        fmap = fmap.view(-1, 1, h, w)  # 1, h, w
        kernel_grid = vutils.make_grid(fmap, normalize=True, scale_each=True, nrow=nrow)  # c, h, w
        writer.add_image('{}_fmap'.format(f_name), kernel_grid, global_step=773031536)

    # ----------------------------------- kernel visualization -----------------------------------
    kernel_vis = 0
    # kernel_vis = 1
    if kernel_vis:
        for n, sub_module in model.named_modules():
            if not isinstance(sub_module, nn.Conv2d):
                continue
            kernels = sub_module.weight
            c_out, c_int, k_w, k_h = tuple(kernels.shape)

            for o_idx in range(c_out):
                kernel_idx = kernels[o_idx, :, :, :].unsqueeze(1)  # 拓展channel
                kernel_grid = vutils.make_grid(kernel_idx, normalize=True, scale_each=True, nrow=c_int)
                writer.add_image('{}_split_in_channel'.format(n), kernel_grid, global_step=o_idx)

            nrow = int(np.sqrt(c_out))
            kernel_all = kernels.view(-1, 1, k_h, k_w)  # 1, h, w
            kernel_grid = vutils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=nrow)  # c, h, w
            writer.add_image('{}_grid'.format(n), kernel_grid, global_step=322)
            print("{}_kernel shape:{}".format(n, tuple(kernels.shape)))

    writer.close()


