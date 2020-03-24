# -*- coding: utf-8 -*-
"""
# @file name  : cifar10_dataset.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-02-29
# @brief      : 构建cifar10 dataset
"""
import os
from PIL import Image
from torch.utils.data import Dataset


class CifarDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        assert (os.path.exists(data_dir)), "data_dir:{} 不存在！".format(data_dir)

        self.data_dir = data_dir
        self._get_img_info()
        self.transform = transform

    def __getitem__(self, index):
        fn, label = self.img_info[index]
        img = Image.open(fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("未获取任何图片路径，请检查dataset及文件路径！")
        return len(self.img_info)

    def _get_img_info(self):
        sub_dir_ = [name for name in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, name))]
        sub_dir = [os.path.join(self.data_dir, c) for c in sub_dir_]

        self.img_info = []
        for c_dir in sub_dir:
            path_img = [(os.path.join(c_dir, i), int(os.path.basename(c_dir))) for i in os.listdir(c_dir) if
                        i.endswith("png")]
            self.img_info.extend(path_img)


