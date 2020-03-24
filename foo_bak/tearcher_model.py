# -*- coding: utf-8 -*-
"""
# @file name  : teacher model.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-02-29
# @brief      : 训练 GhostNet
"""
import matplotlib
matplotlib.use('agg')
import os
import sys
import argparse
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.ghost_net import GhostModule
from models.resnet import resnet56
from tools.cifar10_dataset import CifarDataset
from tools.model_trainer import Teacher
from tools.common_tools import *
from models.lenet import LeNet
from config.config import cfg
from datetime import datetime


if __name__ == '__main__':

    # config
    train_dir = os.path.join(BASE_DIR, "..", "data", "cifar10_train")
    test_dir = os.path.join(BASE_DIR, "..", "data", "cifar10_test")
    # path_checkpoint = "/home/tingsongyu/ghost_net_pytorch/results/03-02_20-22/checkpoint_best.pkl"
    path_checkpoint = "/Users/dream/ai_project/results/03-02_20-22/checkpoint_best.pkl"

    # log dir
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    cfg.log_dir = os.path.join(BASE_DIR, "..", "results", time_str)
    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)

    # ------------------------------------ step 1/5 : 加载数据------------------------------------
    # 构建MyDataset实例
    train_data = CifarDataset(data_dir=train_dir, transform=cfg.transforms_train)
    valid_data = CifarDataset(data_dir=test_dir, transform=cfg.transforms_valid)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=cfg.train_bs, shuffle=True, num_workers=cfg.workers)
    valid_loader = DataLoader(dataset=valid_data, batch_size=cfg.valid_bs, num_workers=cfg.workers)

    # ------------------------------------ step 2/5 : 定义网络------------------------------------
    teacher = resnet56()
    # check_p = torch.load(path_checkpoint, map_location="cpu")
    check_p = torch.load(path_checkpoint, map_location="cpu", encoding='iso-8859-1')
    pretrain_dict = check_p["model_state_dict"]
    state_dict_cpu = state_dict_to_cpu(pretrain_dict)
    teacher.load_state_dict(state_dict_cpu)

    student = resnet56()

    t_map, s_map = [], []

    def t_fmap_hook(m, i, o):
        t_map.append(o)

    def s_fmap_hook(m, i, o):
        s_map.append(o)

    def register_hook(model, hook_func):
        a = 1
        for n, m in model.named_modules():

            if isinstance(m, nn.Conv2d):
                m.register_forward_hook(hook_func)
                print("layer: {} registered".format(n))
                a += 1
                if a > 1:
                    break

    register_hook(teacher, t_fmap_hook)
    register_hook(student, s_fmap_hook)

    fake_img = torch.rand((1, 3, 32, 32))

    # model.to(cfg.device)

    # ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------
    loss_f_cls = nn.CrossEntropyLoss().to(cfg.device)
    loss_f_rec = nn.MSELoss()
    optimizer = optim.SGD(student.parameters(), lr=0.00001, momentum=0.9, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=cfg.factor, patience=cfg.patience, mode='max')
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.factor, milestones=cfg.milestones)

    # ------------------------------------ step 4/5 : 训练 --------------------------------------------------

    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    best_acc = 0
    print("{}\n{}\n{}".format(cfg, loss_f_rec, scheduler))
    for epoch in range(cfg.start_epoch, cfg.max_epoch):

        loss_train, acc_train, mat_train = Teacher.train(train_loader, teacher, student, loss_f_rec, loss_f_cls, optimizer, epoch, t_map, s_map)
        loss_valid, acc_valid, mat_valid = Teacher.valid(valid_loader, model, loss_f)
        print("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} LR:{}".format(
            epoch + 1, cfg.max_epoch, acc_train, acc_valid, loss_train, loss_valid, optimizer.param_groups[0]["lr"]))

        if 'patience' in dir(scheduler):
            scheduler.step(acc_valid)  # ReduceLROnPlateau
        else:
            scheduler.step()            # StepLR

        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
        acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_valid)

        show_confMat(mat_train, cfg.cls_names, "train", cfg.log_dir, verbose=epoch == cfg.max_epoch-1)
        show_confMat(mat_valid, cfg.cls_names, "valid", cfg.log_dir, verbose=epoch == cfg.max_epoch-1)

        plt_x = np.arange(1, epoch+2)
        plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=cfg.log_dir)
        plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir=cfg.log_dir)

        if epoch > (cfg.max_epoch/2) and best_acc < acc_valid:
            best_acc = acc_valid

            checkpoint = {"model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "epoch": epoch,
                      "best_acc": best_acc}

            path_checkpoint = os.path.join(cfg.log_dir, "checkpoint_best.pkl")
            torch.save(checkpoint, path_checkpoint)

    print(" done ~~~~ {}, best acc: {}".format(datetime.strftime(datetime.now(), '%m-%d_%H-%M'), best_acc))



