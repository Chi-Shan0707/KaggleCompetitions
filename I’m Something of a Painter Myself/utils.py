"""
Docstring for utils.py
utils是"utilities"的缩写，意思是“工具”或“实用函数”。
在代码中，它是一个模块文件，提供各种辅助函数和类，用于支持主程序（train.py、test.py）的功能。
"""


import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
# from visdom import Visdom  # Visdom disabled per user request

# No-op Visdom replacement to ensure any visdom calls do nothing
class NoopVisdom:
    def image(self, *args, **kwargs):
        return None
    def line(self, *args, **kwargs):
        return None
    def text(self, *args, **kwargs):
        return None
    def __getattr__(self, name):
        # Return a no-op callable for any other methods
        return lambda *a, **k: None

import numpy as np

def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)
"""
作用：将PyTorch张量转换为NumPy图像数组，便于可视化或保存。
细节：反归一化（从[-1,1]到[0,255]），如果单通道则扩展为3通道（RGB）。
"""

# ==================================================================
class Logger():
    def __init__(self, n_epochs, batches_epoch):
        # Visdom usage has been disabled per user request; use NoopVisdom so all calls are no-ops
        self.viz = NoopVisdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}


    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            # Safely convert tensor losses to Python scalars
            if isinstance(losses[loss_name], torch.Tensor):
                val = losses[loss_name].detach().item()
            else:
                val = float(losses[loss_name])

            if loss_name not in self.losses:
                self.losses[loss_name] = val
            else:
                self.losses[loss_name] += val

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.detach()), opts={'title':image_name})
            else:
                self.viz.image(tensor2image(tensor.detach()), win=self.image_windows[image_name], opts={'title':image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), 
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1
"""
作用：实时记录训练进度、损失和图像（Visdom 已被禁用，不会建立外部连接）。
细节：打印控制台日志（epoch、batch、ETA）、绘制损失曲线和图像窗口（本实现为 no-op）。每个 epoch 结束时更新内部计数。
"""
        
#==================================================================

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))
"""
作用：存储生成的假图像（fake images），在判别器训练时随机采样历史图像，减少模式崩溃，提高GAN稳定性。
细节：缓冲区大小默认50，push_and_pop方法随机替换或返回旧图像。
"""
# =================================================================


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)
"""
作用：自定义学习率调度器，从指定epoch开始线性衰减学习率到0。
细节：step方法返回衰减因子，用于PyTorch的LambdaLR调度器。
"""
# =================================================================

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

"""
作用：对模型层应用正态分布初始化（Conv层权重均值0方差0.02，BatchNorm层权重1方差0.02）。
细节：提高训练收敛速度，避免梯度消失。
"""