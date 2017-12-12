#!/usr/bin/env python

import random

import numpy as np
import torch
from torch.autograd import Variable

import mvtk

from data_loader import get_loader
from solver import Generator
from detect_face import get_faces


if __name__ == '__main__':
    G_path = './stargan_celebA/models/18_12000_G.pth'
    print(G_path)
    G = Generator(64, c_dim=5, repeat_num=6)
    G.load_state_dict(torch.load(G_path))
    G.eval()
    G.cuda()

    for bbox, img_org in get_faces():
        y1, x1, y2, x2 = bbox

        img = img_org[y1:y2, x1:x2]

        xi = img
        # normalize
        xi = xi.astype(np.float32) / 255.
        xi = xi * 2 - 1
        # img -> net input
        xi = xi.transpose(2, 0, 1)
        x = xi[None, :, :, :]
        x = torch.from_numpy(x)
        if torch.cuda.is_available():
            x = x.cuda()
        x = Variable(x, volatile=True)

        c = np.array([[1, 0, 0, 0, 1]], dtype=np.float32)
        c = torch.from_numpy(c)
        if torch.cuda.is_available():
            c = c.cuda()

        y = G(x, c)

        # xi = x[0]
        # xi = xi.data.cpu()
        # xi = (xi + 1) / 2
        # xi = xi.clamp_(0, 1)
        # xi = xi.numpy()
        # xi = (xi * 255).astype(np.uint8)
        # xi = xi.transpose(1, 2, 0)

        yi = y[0]
        yi = yi.data.cpu()
        yi = (yi + 1) / 2
        yi = yi.clamp_(0, 1)
        yi = yi.numpy()
        yi = (yi * 255).astype(np.uint8)
        yi = yi.transpose(1, 2, 0)

        H, W = y2 - y1, x2 - x1
        yi = mvtk.image.resize(yi, height=H, width=W)

        img1 = img_org.copy()
        img2 = img_org.copy()
        img2[y1:y2, x1:x2] = yi

        # mvtk.io.plot_tile([img, yi])
        viz = mvtk.image.tile([img1, img2])
        mvtk.io.plot_tile([img1, img2])
        mvtk.io.show()
