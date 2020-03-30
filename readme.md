
# 目录
项目简介  
依赖  
数据准备  
运行  
>模型训练  
	参数计算  
	特征图可视化  



# 项目简介
本项目对 GhostNet: More Features from Cheap Operations中部分实验进行实践，主要包含三部分内容：
1. 4个模型训练，vgg16， ghost-vgg16， resnet56， ghost-resnet56
2. 参数量计算
3. 特征图可视化  

**更详细解读及实验步骤参见：  
知乎：[GhostNet 解读及代码实验（附代码、超参、日志和与训练模型）](https://zhuanlan.zhihu.com/p/115844245)     
CSDN：[GhostNet 解读及代码实验（附代码、超参和训练日志）](https://blog.csdn.net/u011995719/article/details/105207344)**

# 依赖
Python 3.0+  
PyTorch 1.0+  
tensorboard 2.0.0  
torchstat 0.0.7    

# 数据准备
http://www.cs.toronto.edu/~kriz/cifar.html 下载python版，得到cifar-10-python.tar.gz，解压得到cifar-10-batches-py，并放到  ghostnet\_cifar10/data下，然后执行 python bin/01\_parse\_cifar10\_to\_png.py，可在data/文件夹下获得cifar10\_train 和 cifar10\_test两个文件夹

# 运行
## 模型训练
训练resnet56： python 02\_main.py  -gpu 1 0 -arc resnet56   
训练ghost-resnet56：python bin/02\_main.py -gpu 1 0 -arc resnet56 -replace\_conv    
训练 vgg16：python bin/02\_main.py -gpu 1 0 -arc vgg16    
训练ghost-vgg16：python bin/02\_main.py -gpu 1 0 -arc vgg16 -replace\_conv   

训练完毕将会得到如下表所示精度：
 

|            | Accuray  |  Accuray in paper  |
| :----:     |:----:    | :----:             |
|resnet-56 | 93.4% |93.0%|
|ghost-resnet-56| 91.1%| 92.7%|
|vgg-16 |93.5% |93.6%|
|ghost-vgg-16|92.0% | 93.7%|

## 预训练模型
[ghost-resnet-56 and resnet-56](https://pan.baidu.com/s/10e7CWdHxC18-0pwIr-vXHQ) 密码:uz6f   

[ghost-vgg-16 and vgg-16](https://pan.baidu.com/s/1pnc_Ir5ZwGeSpn9AAx6eZQ) 密码:n82n

## 参数计算
执行 python bin/03\_compute\_flops.py  
需要安装torchstat  
安装方法： pip install torchstat  
torchstat网站：https://github.com/Swall0w/torchstat. 

程序结束可得到下表所示结果   

 
 
|  | Weights|  FLOPs | Weights in paper |  FLOPs in paper| 
|:----: | :----:|:----: | :----: |:----: |
|resnet-56 |0.85M |126M |0.85M |125M|
|ghost-resnet-56| 0.44M| 68M |0.43M| 63M|
|vgg-16| 14.7M |314M| 15M| 313M|
|ghost-vgg-16| 7.4M |160M |7.7M| 158M|


## 特征图可视化

执行 python bin/04\_fmap\_vis.py
在 results/runs下以时间戳为子文件夹记录下events file，然后借助tensorboard就可以查看特征图

来看看VGG16第二个卷积层，是这样的
![](https://github.com/TingsongYu/ghostnet_cifar10/blob/master/data/vgg16-fmap.png)

ghost-vgg16的第二个卷积部分的primary和cheap的卷积特征图如下
![](https://github.com/TingsongYu/ghostnet_cifar10/blob/master/data/ghost-vgg-16_fmap.png)


对Ghost module改进有任何想法的欢迎发邮件一起讨论：yts3221@126.com ，或到博客/知乎中评论   

参考：
ghost部分：https://github.com/huawei-noah/ghostnet   
vgg部分：https://github.com/kuangliu/pytorch-cifar   
resnet部分： https://github.com/akamaster/pytorch_resnet_cifar10   


