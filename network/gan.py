import paddle.nn as nn 
import paddle.nn.functional as F 
from network.base import CNA 


class Generator_GAN(nn.Layer):
    #
    def __init__(self, dim_latent, channel):
        super().__init__()
        self.layers = nn.Sequential(
            CNA(dim_latent, channel*8, 4, 1, 0, 'tconv', 'bn', 'relu'),
            CNA(channel*8, channel*4, 4, 2, 1, 'tconv', 'bn', 'relu'),
            CNA(channel*4, channel*2, 4, 2, 1, 'tconv', 'bn', 'relu'),
            CNA(channel*2, channel, 4, 2, 1, 'tconv', 'bn', 'relu'),
            CNA(channel, 3, 4, 2, 1, 'tconv', False, False)
        )
    #
    def forward(self, x):
        return F.tanh(self.layers(x))


class Discriminator_GAN(nn.Layer):
    #
    def __init__(self, channel):
        super().__init__()
        self.layers = nn.Sequential(
            CNA(3, channel, 4, 2, 1, 'conv', False, 'lrelu'),
            CNA(channel, channel*2, 4, 2, 1, 'conv', 'bn', 'lrelu'),
            CNA(channel*2, channel*4, 4, 2, 1, 'conv', 'bn', 'lrelu'),
            CNA(channel*4, channel*8, 4, 2, 1, 'conv', 'bn', 'lrelu'),
            nn.Conv2D(channel*8, 1, 4, weight_attr=nn.initializer.Normal(0., 0.02), bias_attr=False)
        )
    #
    def forward(self, x):
        return self.layers(x).reshape([-1,1])