import paddle.nn as nn 


class CNA(nn.Layer):
    #
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, conv, norm, act):
        super().__init__()
        if conv == 'conv':
            self.layers = nn.LayerList([nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, weight_attr=nn.initializer.Normal(0., 0.02), bias_attr=False)])
        if conv == 'tconv':
            self.layers = nn.LayerList([nn.Conv2DTranspose(in_channels, out_channels, kernel_size, stride, padding, weight_attr=nn.initializer.Normal(0., 0.02), bias_attr=False)])
        if norm == 'bn':
            self.layers.append(nn.BatchNorm2D(out_channels))
        if act == 'relu':
            self.layers.append(nn.ReLU())
        if act == 'lrelu':
            self.layers.append(nn.LeakyReLU(0.2))
    #
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x