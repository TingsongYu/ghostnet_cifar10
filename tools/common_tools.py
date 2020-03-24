# -*- coding: utf-8 -*-
"""
# @file name  : common_tools.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-02-29
# @brief      : 通用函数
"""
import os
import cv2
import numpy as np
import torch.nn as nn
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances


def img_transform(img_in, transform):
    """
     将img进行预处理，并转换成模型输入所需的形式—— B*C*H*W
    """
    img = img_in.copy()
    img = Image.fromarray(np.uint8(img))
    img = transform(img)
    img = img.unsqueeze(0)    # C*H*W --> B*C*H*W
    return img


def img_preprocess(path_img, mean, std):
    """
    读取图片，转为模型可读的形式
    :return: PIL.image
    """
    assert (os.path.exists(path_img))
    img = cv2.imread(path_img, 1)
    img = cv2.resize(img, (32, 32))
    img = img[:, :, ::-1]   # BGR --> RGB
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    img_input = img_transform(img, transform)
    return img_input


def setup_seed(seed=1):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)     #cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  #cpu/gpu结果一致
        torch.backends.cudnn.benchmark = True   #训练集变化不大时使训练加速


def state_dict_to_cpu(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def replace_conv(model, ghost_module, arc=None, pretrain=False, cheap_pretrian=False, point_conv=False):
    """
    将模型中conv2d层替换为ghost module
    :param model: nn.module, resnet56 or vgg16
    :param ghost_module: GhostModule
    :param arc: str， "resnet56" or "vgg16"
    :param pretrain:  从原始卷积核筛选出，primary卷积核与cheap卷积核并给primary conv赋值
    :param cheap_pretrian: cheap conv是否赋值
    :param point_conv: block中第一个pramary conv是否采用1*1卷积
    :return:
    """
    assert (arc in ("resnet56", "vgg16"))

    model_names = [n for n, k in model.named_modules()]
    for idx, (name, sub_m) in enumerate(model.named_modules()):
        if not isinstance(sub_m, nn.Conv2d):
            continue
        # print(name)
        out_c, in_c, k_s, _ = sub_m.weight.shape
        out_c, in_c, k_s = int(out_c), int(in_c), int(k_s)
        stride = int(sub_m.stride[0])

        if point_conv:
            if "conv1" in name and "layer" in name and name.split(".")[1] == "0":
                print("primary conv {} kernel size is 1*1 ".format(name))
                k_s = 1

        new_conv = ghost_module(in_c, out_c, k_s, stride=stride)

        # 先统一用kaiming初始化
        new_conv.apply(weights_init)

        if pretrain:
            # 聚类，挑选primary卷积核与cheap卷积核
            weights = sub_m.weight.detach()
            kernel_class = kernel_cluster(weights)
            kernel_primary_idx = select_kernel(kernel_class, select_num=int(weights.shape[0] / 2))
            kernel_cheap_idx = list(set(range(out_c)).difference(set(kernel_primary_idx)))
            kernel_order = kernel_primary_idx + kernel_cheap_idx

            # 更新卷积核权值
            # hard code, only for ratio=2(s=2)
            for i, (primary_idx, cheap_idx) in enumerate(zip(kernel_primary_idx, kernel_cheap_idx)):
                new_conv.primary_conv[0].weight.data[i] = torch.nn.Parameter(sub_m.weight[primary_idx])

                # cheap
                if cheap_pretrian:
                    new_conv.cheap_operation[0].weight.data[i] = torch.nn.Parameter(
                        torch.mean(sub_m.weight[cheap_idx], 0).unsqueeze_(0))  # 通道维取均值

            new_conv.fmap_order = kernel_order[:]   # 记录顺序，用于拼接特征图

            # bn
            bn_name = model_names[idx + 1]
            if arc == "resnet56":
                if len(bn_name.split(".")) == 1:
                    bn_pretrain = model.__getattr__(bn_name)
                elif len(bn_name.split(".")) == 3:
                    layer_n, idx, sub_n = bn_name.split(".")
                    bn_pretrain = model._modules[layer_n][int(idx)].__getattr__(sub_n)
            if arc == "vgg16":
                layer_n, idx = bn_name.split(".")
                bn_pretrain = model._modules[layer_n].__getattr__(str(idx))

            # 更新bn参数
            for i, (primary_idx, cheap_idx) in enumerate(zip(kernel_primary_idx, kernel_cheap_idx)):
                new_conv.primary_conv[1].running_mean[i] = bn_pretrain.running_mean[primary_idx]
                new_conv.primary_conv[1].running_var[i] = bn_pretrain.running_var[primary_idx]
                new_conv.primary_conv[1].weight.data[i] = bn_pretrain.weight[primary_idx]
                new_conv.primary_conv[1].bias.data[i] = bn_pretrain.bias[primary_idx]

                # cheap_operation
                if cheap_pretrian:
                    new_conv.cheap_operation[1].running_mean[i] = bn_pretrain.running_mean[cheap_idx]
                    new_conv.cheap_operation[1].running_var[i] = bn_pretrain.running_var[cheap_idx]
                    new_conv.cheap_operation[1].weight.data[i] = bn_pretrain.weight[cheap_idx]
                    new_conv.cheap_operation[1].bias.data[i] = bn_pretrain.bias[cheap_idx]

        # 更新模型
        # print(name)
        if arc == "resnet56":
            if len(name.split(".")) == 1:
                model.add_module(name, new_conv)
            elif len(name.split(".")) == 3:
                layer_n, idx, sub_n = name.split(".")
                # print("name")
                model._modules[layer_n][int(idx)].add_module(sub_n, new_conv)
        if arc == "vgg16":
            name_split = name.split(".")
            model._modules[name_split[0]].add_module(name_split[1], new_conv)

    if not pretrain:
        model.apply(weights_init)

    return model


def replace_conv_entropy(model, ghost_module, arc=None, pretrain=False,
                         cheap_pretrian=False, point_conv=False, bn_ispretrain=False):
    assert (arc in ("resnet56", "vgg16"))

    model_names = [n for n, k in model.named_modules()]
    for idx, (name, sub_m) in enumerate(model.named_modules()):
        if not isinstance(sub_m, nn.Conv2d):
            continue
        # print(name)
        out_c, in_c, k_s, _ = sub_m.weight.shape
        out_c, in_c, k_s = int(out_c), int(in_c), int(k_s)
        stride = int(sub_m.stride[0])

        if point_conv:
            if "conv1" in name and "layer" in name and name.split(".")[1] == "0":
                print("primary conv {} kernel size is 1*1 ".format(name))
                k_s = 1

        new_conv = ghost_module(in_c, out_c, k_s, stride=stride)

        # 先统一用kaiming初始化
        new_conv.apply(weights_init)

        if pretrain:
            # 聚类，挑选primary卷积核与cheap卷积核
            weights = sub_m.weight.detach()

            def sort_kernel(weights, select_num, mode="entropy"):
                assert (mode in ("entropy", "l1"))
                weights_entropy = []
                weights_l1 = []
                for w_idx in range(weights.shape[0]):
                    weights_entropy.append(entropy(weights[w_idx]))
                    weights_l1.append(F.l1_loss(weights[w_idx], torch.zeros_like(weights[w_idx]), reduction="sum"))

                if mode == "entropy":
                    argsort = np.argsort(weights_entropy)
                elif mode == "l1":
                    argsort = np.argsort(weights_l1)

                primary_idx = argsort[select_num:]
                cheap_idx = argsort[:select_num]

                return primary_idx, cheap_idx

            kernel_primary_idx, kernel_cheap_idx = sort_kernel(weights, select_num=int(weights.shape[0] / 2), mode="entropy")
            kernel_order = kernel_primary_idx + kernel_cheap_idx

            # kernel_class = kernel_cluster(weights)
            # kernel_primary_idx = select_kernel(kernel_class, select_num=int(weights.shape[0] / 2))
            # kernel_cheap_idx = list(set(range(out_c)).difference(set(kernel_primary_idx)))
            # kernel_order = kernel_primary_idx + kernel_cheap_idx

            # 更新卷积核权值
            # hard code, only for ratio=2(s=2)
            for i, (primary_idx, cheap_idx) in enumerate(zip(kernel_primary_idx, kernel_cheap_idx)):
                new_conv.primary_conv[0].weight.data[i] = torch.nn.Parameter(sub_m.weight[primary_idx])

                # cheap
                if cheap_pretrian:
                    new_conv.cheap_operation[0].weight.data[i] = torch.nn.Parameter(
                        torch.mean(sub_m.weight[cheap_idx], 0).unsqueeze_(0))  # 通道维取均值

            new_conv.fmap_order = kernel_order[:]   # 记录顺序，用于拼接特征图

            # bn
            if bn_ispretrain:
                bn_name = model_names[idx + 1]
                if arc == "resnet56":
                    if len(bn_name.split(".")) == 1:
                        bn_pretrain = model.__getattr__(bn_name)
                    elif len(bn_name.split(".")) == 3:
                        layer_n, idx, sub_n = bn_name.split(".")
                        bn_pretrain = model._modules[layer_n][int(idx)].__getattr__(sub_n)
                if arc == "vgg16":
                    layer_n, idx = bn_name.split(".")
                    bn_pretrain = model._modules[layer_n].__getattr__(str(idx))

                # 更新bn参数
                for i, (primary_idx, cheap_idx) in enumerate(zip(kernel_primary_idx, kernel_cheap_idx)):
                    new_conv.primary_conv[1].running_mean[i] = bn_pretrain.running_mean[primary_idx]
                    new_conv.primary_conv[1].running_var[i] = bn_pretrain.running_var[primary_idx]
                    new_conv.primary_conv[1].weight.data[i] = bn_pretrain.weight[primary_idx]
                    new_conv.primary_conv[1].bias.data[i] = bn_pretrain.bias[primary_idx]

                    # cheap_operation
                    if cheap_pretrian:
                        new_conv.cheap_operation[1].running_mean[i] = bn_pretrain.running_mean[cheap_idx]
                        new_conv.cheap_operation[1].running_var[i] = bn_pretrain.running_var[cheap_idx]
                        new_conv.cheap_operation[1].weight.data[i] = bn_pretrain.weight[cheap_idx]
                        new_conv.cheap_operation[1].bias.data[i] = bn_pretrain.bias[cheap_idx]

        # 更新模型
        # print(name)
        if arc == "resnet56":
            if len(name.split(".")) == 1:
                model.add_module(name, new_conv)
            elif len(name.split(".")) == 3:
                layer_n, idx, sub_n = name.split(".")
                # print("name")
                model._modules[layer_n][int(idx)].add_module(sub_n, new_conv)
        if arc == "vgg16":
            name_split = name.split(".")
            model._modules[name_split[0]].add_module(name_split[1], new_conv)

    if not pretrain:
        model.apply(weights_init)

    return model


def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)


def entropy(x, n=10):
    x = x.reshape(-1)
    scale = (x.max() - x.min()) / n
    entropy = 0
    for i in range(n):
        p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale), dtype=torch.float) / len(x)
        if p != 0:
            entropy -= p * torch.log(p)

    if entropy == 0:
        print("kernel is : ".format(x))

    return float(entropy.cpu())


def show_confMat(confusion_mat, classes, set_name, out_dir, verbose=False):
    """
    混淆矩阵绘制
    :param confusion_mat:
    :param classes: 类别名
    :param set_name: trian/valid
    :param out_dir:
    :return:
    """
    cls_num = len(classes)
    # 归一化
    confusion_mat_N = confusion_mat.copy()
    for i in range(len(classes)):
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_N, cmap=cmap)
    plt.colorbar()

    # 设置文字
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, list(classes), rotation=60)
    plt.yticks(xlocations, list(classes))
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('Confusion_Matrix_' + set_name)

    # 打印数字
    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 显示

    plt.savefig(os.path.join(out_dir,'Confusion_Matrix' + set_name + '.png'))
    # plt.show()
    plt.close()

    if verbose:
        for i in range(cls_num):
            print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(
                classes[i], np.sum(confusion_mat[i, :]), confusion_mat[i, i],
                confusion_mat[i, i] / (.1 + np.sum(confusion_mat[i, :])),
                confusion_mat[i, i] / (.1 + np.sum(confusion_mat[:, i]))))


def plot_line(train_x, train_y, valid_x, valid_y, mode, out_dir):
    """
    绘制训练和验证集的loss曲线/acc曲线
    :param train_x: epoch
    :param train_y: 标量值
    :param valid_x:
    :param valid_y:
    :param mode:  'loss' or 'acc'
    :param out_dir:
    :return:
    """
    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.ylabel(str(mode))
    plt.xlabel('Epoch')

    location = 'upper right' if mode == 'loss' else 'upper left'
    plt.legend(loc=location)

    plt.title('_'.join([mode]))
    plt.savefig(os.path.join(out_dir, mode + '.png'))
    plt.close()


def get_pair(distan_mat):
    up_mat = np.triu(distan_mat, 1)  # 取下三角矩阵 以及 主对角线
    b = np.argsort(up_mat)  # 按行排序
    bag, pair, bag_b = [], [], []
    for i in range(b.shape[0] - 1):
        if i not in bag and b[i, i + 1] not in bag:  # 已配对的，则跳过
            bag.append(i)
            bag.append(b[i, i + 1])
            pair.append((i, b[i, i + 1]))
    return pair, bag


def kernel_distances(weights):
    w = weights.detach().numpy()
    k_list = []
    for i in range(w.shape[0]):
        k_list.append(w[i].flatten())
    distan_mat = pairwise_distances(k_list, metric="cosine")
    return distan_mat


def kernel_cluster(weights):
    w = weights.detach().numpy()
    k_list = []
    for i in range(w.shape[0]):
        k_list.append(w[i].flatten())

    x = np.array(k_list)
    y_pred = KMeans(n_clusters=int(weights.shape[0] / 2), max_iter=500, random_state=9).fit_predict(x)
    return y_pred


def select_kernel(kernel_class, select_num):
    kernel_retain = []
    for i in range(select_num):
        idx = np.argwhere(kernel_class == i)[0, 0]
        kernel_retain.append(idx)
    return kernel_retain





