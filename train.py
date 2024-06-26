#!/usr/bin/python3

import argparse
import itertools
import multiprocessing

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

import utils
from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from utils import edgedetector
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--resume', action='store_true')
# parser.add_argument('--generator_A2B', type=str, default='output/netG_A2B.pth', help='A2B generator checkpoint file')
# parser.add_argument('--generator_B2A', type=str, default='output/netG_B2A.pth', help='B2A generator checkpoint file')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# 网络定义
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

# cuda
if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

# 是否恢复
if opt.resume:
    netG_A2B.load_state_dict(torch.load('output/netG_A2B.pth', map_location='cuda:0'))
    netG_B2A.load_state_dict(torch.load('output/netG_B2A.pth', map_location='cuda:0'))
    netD_A.load_state_dict(torch.load('output/netD_A.pth', map_location='cuda:0'))
    netD_B.load_state_dict(torch.load('output/netD_B.pth', map_location='cuda:0'))
else:
    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

# 损失函数
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
criterion_edge = torch.nn.L1Loss()

# 优化器和学习率调度器
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                               lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# 输入张量
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)  # 图片
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)  # 标签
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# 数据加载
transforms_ = [transforms.Resize(int(opt.size * 1.12), transforms.InterpolationMode.BICUBIC),  # 该方法保持长宽比
               transforms.RandomCrop(opt.size),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # 图像预处理流程
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_),
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

# 日志输出
logger = Logger(opt.n_epochs, len(dataloader))
# torch优化，输入图像尺寸不同时应该关闭
torch.backends.cudnn.benchmark = True
###################################
# 损失函数权重
identity_loss_weight = 5.0
cycle_loss_weight = 10.0
edge_loss_weight = 10.0
print(f'identity_loss_weight:{identity_loss_weight}  cycle_loss_weight:{cycle_loss_weight}  edge_loss_weight:{edge_loss_weight}')

# 训练
multiprocessing.freeze_support()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # 设置模型输入
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        ######  生成器  ######
        optimizer_G.zero_grad()

        # 一致性损失       默认乘5.0倍率
        # G_A2B输入B时，输出应该和输入保持一致
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B) * identity_loss_weight

        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A) * identity_loss_weight

        # GAN 损失        默认1.0
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # 循环一致损失      默认乘10.0倍率
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * cycle_loss_weight

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * cycle_loss_weight

        # 边缘损失**
        edge_real_A = edgedetector(real_A, 0)
        edge_fake_B = edgedetector(fake_B, 0)
        loss_edge_A2B = criterion_edge(edge_fake_B, edge_real_A) * edge_loss_weight
        # print(loss_edge_A2B.item())
        # Image.fromarray(utils.tensor2image(real_A)).show()
        # Image.fromarray(utils.tensor2image(edge_real_A)).show()
        # Image.fromarray(utils.tensor2image(edge_fake_B.clone().detach())).show()

        # opencv方法，无法传递梯度
        # edge_real_A = edgedetector(real_A, 0)
        # edge_fake_B_nograd = edgedetector(fake_B.clone().detach(), 1)
        # edge_fake_B = fake_B.clone().copy_(edge_fake_B_nograd)  # 保持梯度  ******用copy?
        # loss_edge_A2B = criterion_edge(edge_fake_B, edge_real_A)
        # print(loss_edge_A2B.item())

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_edge_A2B
        loss_G.backward()

        optimizer_G.step()
        ###################################

        ###### 判别器 A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### 判别器 B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################

        # Progress report (http://localhost:8097)
        logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B),
                    'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B), 'loss_edge_A2B': loss_edge_A2B},
                   images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), 'output/netG_B2A.pth')
    torch.save(netD_A.state_dict(), 'output/netD_A.pth')
    torch.save(netD_B.state_dict(), 'output/netD_B.pth')
###################################
