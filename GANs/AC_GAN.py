import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import torch.nn.init as init


class Generator(nn.Module):
    def __init__(self, noise_std):
        super(Generator, self).__init__()
        self.noise_std = noise_std
        self.fc1 = nn.Linear(110, 384)

        def transposed_convolution(in_channels, out_channels, kernel_size, stride,
                                   padding, output_padding, bn=True, relu=True):
            block = nn.Sequential()
            block.add_module('conv',
                             nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding,
                                                bias=False))
            if bn:
                block.add_module('batchNorm', nn.BatchNorm2d(out_channels))
            if relu:
                block.add_module('relu', nn.ReLU(inplace=True))
            else:
                block.add_module('tanh', nn.Tanh())
            return block

        self.tc1 = transposed_convolution(384, 192, 4, 1, 0, 0)
        self.tc2 = transposed_convolution(192, 96, 4, 2, 1, 0)
        self.tc3 = transposed_convolution(96, 48, 4, 2, 1, 0)
        self.tc4 = transposed_convolution(48, 3, 4, 2, 1, 0, False, False)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(x.shape[0], 384, 1, 1)
        x = self.tc1(x)
        x = self.tc2(x)
        x = self.tc3(x)
        x = self.tc4(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def convolution(in_channels, out_channels, kernel_size, stride,
                        padding, bn=True):
            block = nn.Sequential()
            block.add_module('conv',
                             nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
            if bn:
                block.add_module('batchNorm', nn.BatchNorm2d(out_channels))
            block.add_module('leaky_relu', nn.LeakyReLU(0.2, inplace=True))
            block.add_module('dropout', nn.Dropout(p=0.5, inplace=False))
            return block

        self.conv1 = convolution(3, 16, 3, 2, 1, False)
        self.conv2 = convolution(16, 32, 3, 1, 1)
        self.conv3 = convolution(32, 64, 3, 2, 1)
        self.conv4 = convolution(64, 128, 3, 1, 1)
        self.conv5 = convolution(128, 256, 3, 2, 1)
        self.conv6 = convolution(256, 512, 3, 1, 1)
        self.fc_dis = nn.Linear(8192, 1)
        self.fc_cla = nn.Linear(8192, 10)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(-1, 8192)
        res1 = self.fc_dis(x)
        res2 = self.fc_cla(x)
        real_or_fake = self.sigmoid(res1).view(-1, 1)
        predict_classes = self.softmax(res2)
        return real_or_fake, predict_classes


def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
    elif hasattr(m, 'weight') and (classname.find('BatchNorm') != -1):
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(m.bias.data, 0.0)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    generator = Generator(0.1)
    generator = generator.to(device)
    summary(generator, (110,), 5)

    discriminator = Discriminator()
    discriminator = discriminator.to(device)
    summary(discriminator, (3, 32, 32), 5)
