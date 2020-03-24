# -*- coding: utf-8 -*-
"""
# @file name  : main.py
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
from models.ghost_net import GhostModule
from models.resnet import resnet56
from models.vgg import VGG
from tools.cifar10_dataset import CifarDataset
from tools.model_trainer import ModelTrainer
from tools.common_tools import *
from config.config import cfg
from datetime import datetime

print('PID:{}'.format(os.getpid()))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Ghostnet Training')
    parser.add_argument('-bs', type=int, default=128)
    parser.add_argument('-max_epoch', type=int, default=190)
    parser.add_argument('-start_epoch', type=int, default=0)
    parser.add_argument('-lr', type=float, default=0.1)
    parser.add_argument('-gpu', type=int, nargs='+')
    parser.add_argument('-pretrain', default=False, action='store_true')
    parser.add_argument('-frozen_primary', default=False, action='store_true')
    parser.add_argument('-point_conv', default=False, action='store_true')
    parser.add_argument('-replace_conv', default=False, action='store_true')
    parser.add_argument('-low_lr', default=False, action='store_true')
    parser.add_argument('-arc', type=str, default="resnet56", help="architecture, support resnet56/vgg16  only")
    args = parser.parse_args()

    # gpu设置
    Isparallel = True if len(args.gpu) > 1 else False
    gpu_list_str = ','.join(map(str, args.gpu))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)

    print("set gpu list :{}".format(gpu_list_str))
    device_count = torch.cuda.device_count()
    print("\ndevice_count: {}".format(device_count))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # path
    train_dir = os.path.join(BASE_DIR, "..", "data", "cifar10_train")
    test_dir = os.path.join(BASE_DIR, "..", "data", "cifar10_test")

    path_checkpoint = "path to pretrain model/checkpoint_best.pkl"

    # log dir
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join(BASE_DIR, "..", "results", time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # random seed
    setup_seed(1)

    # ------------------------------------ step 1/5 : 加载数据------------------------------------
    # 构建MyDataset实例
    train_data = CifarDataset(data_dir=train_dir, transform=cfg.transforms_train)
    valid_data = CifarDataset(data_dir=test_dir, transform=cfg.transforms_valid)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=args.bs, shuffle=True, num_workers=cfg.workers)
    valid_loader = DataLoader(dataset=valid_data, batch_size=args.bs, num_workers=cfg.workers)

    # ------------------------------------ step 2/5 : 定义网络------------------------------------

    if args.arc == "vgg16":
        model = VGG("VGG16")
    elif args.arc == "resnet56":
        model = resnet56()
    else:
        raise ValueError("{} is not define!".format(args.arc))

    # check_p = torch.load(path_checkpoint, map_location="cpu", encoding='iso-8859-1')
    if args.pretrain:
        check_p = torch.load(path_checkpoint, map_location="cpu")
        pretrain_dict = check_p["model_state_dict"]
        state_dict_cpu = state_dict_to_cpu(pretrain_dict)
        model.load_state_dict(state_dict_cpu)
        print("load state dict from :{} done~~".format(path_checkpoint))

    # ghost-resnet
    if args.replace_conv:
        model = replace_conv(model, GhostModule, arc=args.arc,
                             pretrain=args.pretrain, cheap_pretrian=False, point_conv=args.point_conv)
        # model = replace_conv_entropy(model, GhostModule, arc=args.arc,
        #                      pretrain=args.pretrain, cheap_pretrian=False, point_conv=args.point_conv)
        print("repalce all conv layer to ghost module")
        print("model architecture: ", model)

    if args.frozen_primary:
        for n, p in model.named_parameters():
            if "primary_conv" in n:
                p.requires_grad = False
                print("{} is not requires_grad".format(n))

    if Isparallel and torch.cuda.is_available():
        model = torch.nn.DataParallel(model)

    model.to(device)

    # ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------

    loss_f = nn.CrossEntropyLoss().to(device)
    if args.low_lr:
        primary_params_id = [id(p) for n, p in model.named_parameters() if "primary" in n]
        primary_params = filter(lambda p: id(p) in primary_params_id, model.parameters())
        base_params = filter(lambda p: id(p) not in primary_params_id, model.parameters())
        optimizer = optim.SGD([
                {'params': primary_params, 'lr': 0.1 * args.lr},
                {'params': base_params}],
                lr=args.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    # if args.pretrain:
    #     scheduler = ReduceLROnPlateau(optimizer, factor=cfg.factor, patience=cfg.patience, mode='max')
    # else:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.factor, milestones=cfg.milestones)

    # ------------------------------------ step 4/5 : 训练 --------------------------------------------------

    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    best_acc, best_epoch = 0, 0

    print("args:\n{}\n cfg:\n{}\n loss_f:\n{}\n scheduler:\n{}\n optimizer:\n{}".format(
        args, cfg, loss_f, scheduler, optimizer))

    for epoch in range(args.start_epoch, args.max_epoch):

        loss_train, acc_train, mat_train = ModelTrainer.train(train_loader, model, loss_f, optimizer, epoch, device, args)
        loss_valid, acc_valid, mat_valid = ModelTrainer.valid(valid_loader, model, loss_f, device)
        print("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} LR:{}".format(
            epoch + 1, args.max_epoch, acc_train, acc_valid, loss_train, loss_valid, optimizer.param_groups[0]["lr"]))

        if 'patience' in dir(scheduler):
            scheduler.step(acc_valid)  # ReduceLROnPlateau
        else:
            scheduler.step()            # StepLR

        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
        acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_valid)

        show_confMat(mat_train, cfg.cls_names, "train", log_dir, verbose=epoch == args.max_epoch-1)
        show_confMat(mat_valid, cfg.cls_names, "valid", log_dir, verbose=epoch == args.max_epoch-1)

        plt_x = np.arange(1, epoch+2)
        plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
        plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir=log_dir)

        if epoch > (args.max_epoch/2) and best_acc < acc_valid:
            best_acc = acc_valid
            best_epoch = epoch

            checkpoint = {"model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "epoch": epoch,
                      "best_acc": best_acc}

            path_checkpoint = os.path.join(log_dir, "checkpoint_best.pkl")
            torch.save(checkpoint, path_checkpoint)

    print(" done ~~~~ {}, best acc: {} in :{}".format(datetime.strftime(datetime.now(), '%m-%d_%H-%M'), 
                                                      best_acc, best_epoch))



