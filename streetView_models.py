import torch
import torch.nn as nn
import torch.nn.functional as F

class Gnet(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 1024, 1, 1, bias=False),#1024x1x1
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 256, 8, 1, bias=False),#256x8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),#128x16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),#64x32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, (3, 4), (1, 2), 1, bias=False),#3x64*128
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.main(x)
        return output

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class FrontEnd(nn.Module):
    ''' front end part of discriminator and Q'''

    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 64, (3, 4), (1, 2), (1, 1)), #64x32x32
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),#128x16x16
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),#256x8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 1024, 8, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        output = self.main(x)
        return output


class Dnet(nn.Module):

    def __init__(self):
        super().__init__()
    
        self.main = nn.Sequential(
            nn.Conv2d(1024, 1, 1),
            nn.Sigmoid()
        )
    

    def forward(self, x):
        output = self.main(x).view(-1, 1)
        return output


class Qnet(nn.Module):

    def __init__(self, z_dis, z_con):
        super().__init__()

        self.conv = nn.Conv2d(1024, 128, 1, bias=False)
        self.bn = nn.BatchNorm2d(128)
        self.lReLU = nn.LeakyReLU(0.1, inplace=True)
        self.conv_disc = nn.Conv2d(128, z_dis, 1)
        self.conv_mu = nn.Conv2d(128, z_con, 1)
        self.conv_var = nn.Conv2d(128, z_con, 1)

    def forward(self, x):

        y = self.conv(x)

        disc_logits = self.conv_disc(y).squeeze()

        mu = self.conv_mu(y).squeeze()
        var = self.conv_var(y).squeeze().exp()

        return disc_logits, mu, var 

if __name__ == "__main__":
    x = torch.Tensor(5,3,32,64)
    print(x.shape)
    front = FrontEnd()
    print(front)
    # qnet = Qnet()
    # dnet = Dnet()
    z = front(x)
    # d = dnet(z)
    print(z.shape)
    # print(d)
    