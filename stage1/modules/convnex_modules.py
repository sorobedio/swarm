import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ConvNeXt_Base_Weights, convnext_base

import torch
super_net_name = "ofa_supernet_mbv3_w10"
# other options:
#    ofa_supernet_resnet50 /
#    ofa_supernet_mbv3_w12 /
#    ofa_supernet_proxyless

super_net = torch.hub.load('mit-han-lab/once-for-all', super_net_name, pretrained=True).eval()

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

class Activation_fn(nn.Module):
    def __init__(self):
        super(Activation_fn, self).__init__()
    def forward(self,x):
        x = x*torch.sigmoid(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, z_channels, z_ch=1, input_size=32, in_dim=16032, ch=8, out_dim=16384,
                 z_size=32, double_z=True):
        super(Encoder, self).__init__()
        self.in_channels = in_channels

        self.input_size=input_size
        self.z_ch = z_ch
        self.ch =  ch
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.z_size =  z_size
        self.double_z=double_z
        if self.double_z:
            z_channels = z_channels*2
        self.z_channels = z_channels

        weights = ConvNeXt_Base_Weights.DEFAULT
        model = convnext_base(weights=weights)
        model.features[0][0]=nn.Conv2d(in_channels, 128, kernel_size=4, stride=4)
        # self.features = model.features
        # self.encoder = nn.Sequential(
        #     model.features,
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),
        # )
        self.conv = nn.Conv2d(z_ch, z_channels, kernel_size=3, stride=1, padding=1)
        self.encoder =  model.features
        self.fc_in = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        x = x.reshape(-1, self.ch, self.in_dim)
        x = self.fc_in(x)
        x = x.reshape((-1, self.in_channels, self.input_size, self.input_size))
        z = self.encoder(x)
        # print(z.shape)
        z = z.reshape(-1, self.z_ch, self.z_size, self.z_size)

        z = self.conv(z)
        # print(z.shape)
        return z

class Decoder(nn.Module):
    def __init__(self, in_channels, z_channels, z_ch=1, input_size=32, in_dim=16032, ch=8, out_dim=16384,
                 z_size=32, double_z=True):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.double_z = double_z

        self.input_size = input_size
        self.z_ch = z_ch
        self.ch = ch
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.z_size = z_size
        if self.double_z:
            z_channels = z_channels * 2
        self.z_channels = z_channels
        self.fc_out = nn.Linear(out_dim, in_dim, bias=False)
        self.conv = nn.Conv2d(z_channels, z_ch, kernel_size=3, stride=1, padding=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d( 1024, 512, 4, stride=2, padding=1),
            Activation_fn(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            Activation_fn(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            Activation_fn(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            Activation_fn(),
            nn.ConvTranspose2d(64, in_channels, 4, stride=2, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            Activation_fn(),
        )
    def forward(self,z):
        z = self.conv(z)
        z = z.reshape(-1, 1024, self.input_size//32, self.input_size//32)
        x = self.decoder(z)
        # print('=================================')
        # print(x.shape)
        # print('--------------------------')
        x = x.reshape(-1, self.ch, self.out_dim)
        x = self.fc_out(x)
        return x











