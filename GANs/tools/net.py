import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.ct1 = nn.ConvTranspose2d(latent_dim, 512, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.ac1 = nn.LeakyReLU(0.2, True)
        
        self.ct2 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.ac2 = nn.LeakyReLU(0.2, True)
        
        self.ct3 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.ac3 = nn.LeakyReLU(0.2, True)
        
        self.ct4 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.ac4 = nn.LeakyReLU(0.2, True)
        
        self.ct5 = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)
        self.ac5 = nn.Tanh()
    
    def forward(self, x):
        out = self.ac1(self.bn1(self.ct1(x)))
        out = self.ac2(self.bn2(self.ct2(out)))
        out = self.ac3(self.bn3(self.ct3(out)))
        out = self.ac4(self.bn4(self.ct4(out)))
        out = self.ac5(self.ct5(out))
        return out
    
    
class Discriminator(nn.Module):
    def __init__(self, WGAN=False):
        super(Discriminator, self).__init__()
        self.WGAN = WGAN
        
        self.cv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.ac1 = nn.LeakyReLU(0.2, True)
        self.cv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.ac2 = nn.LeakyReLU(0.2, True)
        self.cv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.ac3 = nn.LeakyReLU(0.2, True)
        self.cv4 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.ac4 = nn.LeakyReLU(0.2, True)
        self.cv5 = nn.Conv2d(512, 1, 4, bias=False)
        if not WGAN:
            self.ac5 = nn.Sigmoid()
        
    def forward(self, x):
        out = self.ac1(self.bn1(self.cv1(x)))
        out = self.ac2(self.bn2(self.cv2(out)))
        out = self.ac3(self.bn3(self.cv3(out)))
        out = self.ac4(self.bn4(self.cv4(out)))
        out = self.cv5(out)
        if not self.WGAN:
            out = self.ac5(out)
        return out