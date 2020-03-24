# -*- coding: utf-8 -*-
"""
# @file name  : model_trainer.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-02-29
# @brief      : 模型训练类
"""
import torch
import time
import numpy as np
from functools import reduce
from config.config import cfg


class ModelTrainer(object):

    @staticmethod
    def train(data_loader, model, loss_f, optimizer, epoch_id, device, args):
        model.train()

        conf_mat = np.zeros((cfg.class_num, cfg.class_num))
        loss_sigma = []

        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            optimizer.zero_grad()
            loss = loss_f(outputs, labels)
            loss.backward()
            optimizer.step()

            # 统计预测信息
            _, predicted = torch.max(outputs.data, 1)

            # 统计混淆矩阵
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.

            # 统计loss
            loss_sigma.append(loss.item())
            acc_avg = conf_mat.trace() / conf_mat.sum()

            # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
            if i % cfg.log_interval == cfg.log_interval - 1:
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch_id + 1, args.max_epoch, i + 1, len(data_loader), np.mean(loss_sigma), acc_avg))

        return np.mean(loss_sigma), acc_avg, conf_mat

    @staticmethod
    def valid(data_loader, model, loss_f, device):
        model.eval()

        conf_mat = np.zeros((cfg.class_num, cfg.class_num))
        loss_sigma = []

        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = loss_f(outputs, labels)

            # 统计预测信息
            _, predicted = torch.max(outputs.data, 1)

            # 统计混淆矩阵
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.

            # 统计loss
            loss_sigma.append(loss.item())

        acc_avg = conf_mat.trace() / conf_mat.sum()

        return np.mean(loss_sigma), acc_avg, conf_mat


class Teacher(object):

    @staticmethod
    def train(data_loader, teacher, student, loss_f_rec, loss_f_cls, optimizer, epoch_id, t_fmap, s_fmap):
        teacher.eval()
        student.train()

        conf_mat = np.zeros((cfg.class_num, cfg.class_num))
        loss_sigma = []

        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)

            t_o = teacher(inputs)
            s_o = student(inputs)

            optimizer.zero_grad()
            loss = loss_f_rec(t_o, s_o)

            loss_list = [loss_f_rec(t, s) for (t, s) in zip(t_fmap, s_fmap)]
            loss_rec = reduce(lambda x, y: x+y, loss_list)
            loss_rec = loss_rec * 0.0001

            loss_cls = loss_f_cls(s_o, labels)

            loss_lambda = 0.9

            loss_sum = loss_lambda*loss_cls + (1-loss_lambda)*loss_rec
            print("cls loss:{} rec loss:{}".format(loss_cls, loss_rec))

            loss.backward(retain_graph=True)

            optimizer.step()

            # 统计预测信息
            _, predicted = torch.max(s_o.data, 1)

            # 统计混淆矩阵
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.

            # 统计loss
            loss_sigma.append(loss.item())
            acc_avg = conf_mat.trace() / conf_mat.sum()

            # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
            # if i % cfg.log_interval == cfg.log_interval - 1:
            if 1:
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch_id + 1, cfg.max_epoch, i + 1, len(data_loader), np.mean(loss_sigma), acc_avg))

        return np.mean(loss_sigma), acc_avg, conf_mat

    @staticmethod
    def valid(data_loader, model, loss_f):
        model.eval()

        conf_mat = np.zeros((cfg.class_num, cfg.class_num))
        loss_sigma = []

        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)

            outputs = model(inputs)
            loss = loss_f(outputs, labels)

            # 统计预测信息
            _, predicted = torch.max(outputs.data, 1)

            # 统计混淆矩阵
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.

            # 统计loss
            loss_sigma.append(loss.item())

        acc_avg = conf_mat.trace() / conf_mat.sum()

        return np.mean(loss_sigma), acc_avg, conf_mat


