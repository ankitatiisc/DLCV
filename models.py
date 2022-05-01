import torch
import torch.nn as nn



# Generator architecture to be used for training
class Generator(nn.Module):
    def __init__(self, nz, ndim, nc):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ndim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ndim * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ndim * 8, ndim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndim * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ndim * 4, ndim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndim * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ndim * 2, ndim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndim),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ndim, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.layers(input)

# Disctiminator architecture used for training 
class Discriminator(nn.Module):
    def __init__(self, nc, ndim):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndim, ndim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndim * 2, ndim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndim * 4, ndim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.layers(input)