#!/usr/bin/env python

import random

import numpy as np
import torch
from torch.autograd import Variable

import mvtk

from data_loader import get_loader
from solver import Generator


if __name__ == '__main__':
    G_path = './stargan_celebA/models/20_4000_G.pth'
    print(G_path)
    G = Generator(64, c_dim=5, repeat_num=6)
    G.load_state_dict(torch.load(G_path))
    G.eval()
    G.cuda()

    random.seed(1)
    data_loader = get_loader(
        image_path='./data/CelebA_nocrop/images',
        metadata_path='./data/list_attr_celeba.txt',
        crop_size=178,
        image_size=128,
        batch_size=16,
        dataset='CelebA',
        mode='test',
    )
    for i, (real_x, org_c) in enumerate(data_loader):
        if torch.cuda.is_available():
            real_x = real_x.cuda()
        real_x = Variable(real_x, volatile=True)

        target_c = org_c.clone()
        for c in target_c:
            c[0], c[1], c[2] = 1, 0, 0  # Hair: black
            c[3] = 0  # Gender: 0=female, 1=male
            c[4] = 1  # Aged: 0=aged, 1=young
        if torch.cuda.is_available():
            target_c = target_c.cuda()
        target_c = Variable(target_c, volatile=True)

        fake_x = G(real_x, target_c)

        real_x = real_x.data.cpu()
        fake_x = fake_x.data.cpu()

        real_x = (real_x + 1) / 2
        real_x = real_x.clamp_(0, 1)
        real_x = real_x.numpy()
        real_x = (real_x * 255).astype(np.uint8)

        fake_x = (fake_x + 1) / 2
        fake_x = fake_x.clamp_(0, 1)
        fake_x = fake_x.numpy()
        fake_x = (fake_x * 255).astype(np.uint8)

        vizs = []
        for xi_r, xi_f in zip(real_x, fake_x):
            xi_r = xi_r.transpose(1, 2, 0)
            xi_f = xi_f.transpose(1, 2, 0)
            viz = np.hstack([xi_r, xi_f])
            vizs.append(viz)

        mvtk.io.plot_tile(vizs)
        mvtk.io.show()

        break
