# -*- coding: utf-8 -*-
"""
# @file name  : inference_in_test.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-02-29
# @brief      : 测试test数据集上指标
"""
import matplotlib
matplotlib.use('agg')
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

# sys.path.append(os.path.abspath("../../"))
# sys.path.append("/home/tingsongyu/ghost_net_pytorch")
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import vgg16_bn, resnet50
#from models.densenet import DenseNet121
#from models.lenet import LeNet
from models.ghost_net import GhostModule
from models.vgg import VGG
from models.resnet import resnet56
from tools.cifar10_dataset import CifarDataset
from tools.model_trainer import ModelTrainer
from tools.common_tools import *
from config.config import cfg
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':

    test_dir = os.path.join(BASE_DIR, "..", "data", "cifar10_test")
    path_checkpoint = os.path.join(BASE_DIR, "..", "..", "results/03-02_20-22/checkpoint_best.pkl")  # resnet-image
    path_checkpoint = os.path.join(BASE_DIR, "..", "..", "results/03-06_16-54/checkpoint_best.pkl")  # vgg-cifar
    valid_data = CifarDataset(data_dir=test_dir, transform=cfg.transforms_valid)
    valid_loader = DataLoader(dataset=valid_data, batch_size=cfg.valid_bs, num_workers=cfg.workers)
    log_dir = "../../results"

    model = resnet56()
    model = VGG("VGG16")


    check_p = torch.load(path_checkpoint, map_location="cpu", encoding='iso-8859-1')
    pretrain_dict = check_p["model_state_dict"]
    print("best acc: {} in epoch:{}".format(check_p["best_acc"], check_p["epoch"]))
    state_dict_cpu = state_dict_to_cpu(pretrain_dict)
    model.load_state_dict(state_dict_cpu)

    # resnet --> ghost-resnet
    model = replace_conv(model, GhostModule, arc="vgg16", pretrain=False)
    # model = replace_conv(model, GhostModule, pretrain=True)

    Isparallel = False
    if Isparallel and torch.cuda.is_available():
        model = torch.nn.DataParallel(model)

    model.to(device)

    loss_f = nn.CrossEntropyLoss()

    loss_valid, acc_valid, mat_valid = ModelTrainer.valid(valid_loader, model, loss_f, device)

    show_confMat(mat_valid, cfg.cls_names, "valid", log_dir, verbose=True)

    print("dataset: {}, acc: {} loss: {}".format(test_dir, acc_valid, loss_valid))

