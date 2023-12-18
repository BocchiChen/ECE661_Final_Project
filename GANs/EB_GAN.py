import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.nn.init as init


class Generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=3, input_size=32):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * 8 * 8),
            nn.BatchNorm1d(128 * 8 * 8),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                init.normal_(m.weight.data, 0, 0.02)
                if m.bias is not None:
                    init.constant_(m.bias.data, 0)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, 8, 8)
        x = self.deconv(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, input_size=32):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.ReLU(),
        )
        self.nv = nn.Sequential(
            nn.Linear(64 * 16 * 16, 32),
        )
        self.fc = nn.Sequential(
            nn.Linear(32, 64 * 16 * 16),
            nn.BatchNorm1d(64 * 16 * 16),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Sigmoid(),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                init.normal_(m.weight.data, 0, 0.02)
                if m.bias is not None:
                    init.constant_(m.bias.data, 0)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(x.size()[0], -1)
        out = self.nv(x)
        x = self.fc(out)
        x = x.view(-1, 64, 16, 16)
        x = self.deconv(x)

        return x, out


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    generator = Generator()
    generator = generator.to(device)
    summary(generator, (100,), 5)

    discriminator = Discriminator()
    discriminator = discriminator.to(device)
    summary(discriminator, (3, 32, 32), 5)