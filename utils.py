import random
import time
import datetime
import sys
from math import ceil
from os import makedirs

from torch.autograd import Variable
import torch.nn.functional as F
import torch
# from visdom import Visdom
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2


def tensor2image(tensor):
    image = 127.5 * (tensor[0].cpu().float().numpy() + 1.0)  # 从[-1,1]转换为[0,2]再转换为[0,255]
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))  # 灰度图复制为3个通道
    image = np.transpose(image, (1, 2, 0))  # 转为HWC的顺序
    return image.astype(np.uint8)


def image2tensor(image: np.ndarray):
    if image.shape[0] == 1 or len(image.shape) == 2:  # 灰度图
        image = np.tile(image, (3, 1, 1))  # 灰度图复制为3个通道 (3,256,256)
    elif len(image.shape) == 3:
        image = np.transpose(image, (2, 0, 1))
    else:
        raise TypeError

    image = image / 127.5 - 1.0  # 归一化
    img_tensor = torch.from_numpy(image).unsqueeze(0).to('cuda:0')
    return img_tensor


def edgedetector(img_tensor: torch.Tensor, img_type: int):
    img_tensor = torch.mean(img_tensor, dim=1, keepdim=True)  # 转单通道灰度图
    # Image.fromarray(tensor2image(img_tensor)).show()

    # 创建 Sobel 算子核心部分（水平方向）
    sobel_x_core = torch.tensor([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]], dtype=torch.float32)

    sobel_x = sobel_x_core.unsqueeze(0).unsqueeze(0).to('cuda:0')

    # 创建 Sobel 算子核心部分（垂直方向）
    sobel_y_core = torch.tensor([[1, 2, 1],
                                 [0, 0, 0],
                                 [-1, -2, -1]], dtype=torch.float32)

    sobel_y = sobel_y_core.unsqueeze(0).unsqueeze(0).to('cuda:0')

    sobel_x_output = F.conv2d(img_tensor, sobel_x, padding=1)
    sobel_y_output = F.conv2d(img_tensor, sobel_y, padding=1)
    edges = torch.sqrt(sobel_x_output**2 + sobel_y_output**2)
    # 计算边缘张量的最大值和最小值
    max_value = edges.max()
    min_value = edges.min()

    # 将边缘张量重新缩放到 [-1, 1] 的范围内
    edges_normalized = 2 * (edges - min_value) / (max_value - min_value) - 1

    # Image.fromarray(tensor2image(edges)).show()
    # opencv版
    # image = tensor2image(img_tensor)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    # if img_type == 0:  # 原图real
    #     sobelx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3, scale=0.5)  # 原图再轻一点？
    #     sobely = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3, scale=0.5)
    # elif img_type == 1:  # 生成图fake
    #     sobelx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3, scale=1)
    #     sobely = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3, scale=1)
    # else:
    #     raise ValueError
    #
    # edges = cv2.magnitude(sobelx, sobely)
    # edges = cv2.Canny(image, 100, 200)  # 参数分别为低阈值和高阈值
    # Image.fromarray(edges).show()
    return edges_normalized


class Logger:
    def __init__(self, n_epochs, batches_epoch):
        # self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.losses_history: {str: list} = {}
        self.loss_windows = {}
        self.image_windows = {}
        makedirs('./output/train_output', exist_ok=True)

    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write(
            '\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            if (i + 1) == len(losses.keys()):  # 如果遍历到最后一个
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name] / self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name] / self.batch))

        batches_done = self.batches_epoch * (self.epoch - 1) + self.batch
        batches_left = self.batches_epoch * (self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left * self.mean_period / batches_done)))

        # 每十代保存一次图片结果
        if self.epoch % 10 == 0 and self.batch == 1:
            for image_name, tensor in images.items():
                # 使用visdom服务器输出训练信息时
                # if image_name not in self.image_windows:
                # self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title':image_name})
                # else:
                # self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})
                # 直接输出到文件时
                Image.fromarray(tensor2image(tensor.data)).save(
                    f'./output/train_output/{image_name}_' + f'epoch{self.epoch}.jpg')
        # epoch结束时保存损失函数图像
        if (self.batch % self.batches_epoch) == 0:
            # 保存损失函数历史
            for loss_name, loss in self.losses.items():
                # 文件损失函数图输出
                if loss_name not in self.losses_history:
                    self.losses_history[loss_name] = [loss]
                else:
                    self.losses_history[loss_name].append(loss)
                # visdom输出
                # if loss_name not in self.loss_windows:
                #     # self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]),
                #     #                                                 opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                #     pass
                # else:
                #     # self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                #     pass
                # 为下一代重置loss
                self.losses[loss_name] = 0.0

            # 绘制图像
            loss_fig, loss_ax = plt.subplots(ceil(len(self.losses_history) / 3), 3)
            loss_ax = loss_ax.reshape(-1)  # 二维变一维
            for i, (loss_name, loss) in enumerate(self.losses_history.items()):
                loss_ax[i].plot(range(len(loss)), loss)
                loss_ax[i].set_xlabel('Epoch')
                loss_ax[i].set_ylabel(loss_name)
                loss_ax[i].set_title(loss_name)
            loss_fig.tight_layout()  # 调整布局
            plt.savefig('./output/train_output/losses_curve.jpg')
            plt.close('all')
            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1


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
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
