import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import ConvNeXt_Base_Weights, convnext_base
import clip
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
    def __init__(self, in_channels, z_channels, z_ch=4, ch=8,
                 z_size=16, z_dim=512, double_z=True):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.z_ch = z_ch
        self.ch =  ch
        self.z_dim=z_dim
        self.z_size =  z_size
        self.double_z=double_z
        if self.double_z:
            z_channels = z_channels*2
        self.z_channels = z_channels

        # weights = ConvNeXt_Base_Weights.DEFAULT
        # model = convnext_base(weights=weights)
        model, _ = clip.load("ViT-B/32", device='cuda')
        for param in model.parameters():
            param.requires_grad = True
        self.encoder=  model.encode_image


        self.conv = nn.Conv2d(z_ch, z_channels, kernel_size=3, stride=1, padding=1)
        # self.encoder =  model.features
        self.fc_in = nn.Linear(512, 1024, bias=False)

    def forward(self, x):
        z = self.encoder(x).squeeze()
        z= self.fc_in(z)
        # print(z.shape)
        z = z.reshape(-1, self.z_ch, self.z_size, self.z_size)
        z = self.conv(z)
        # print(z.shape)
        return z

class Decoder(nn.Module):
    def __init__(self, in_channels, z_channels, z_ch=4, ch=4, z_dim=512,
                 z_size=16, double_z=True):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.double_z = double_z
        self.z_dim=z_dim

        self.z_ch = z_ch
        self.ch = ch
        self.z_size = z_size
        if self.double_z:
            z_channels = z_channels * 2
        self.z_channels = z_channels
        self.decoder = nn.Sequential(
            nn.Conv2d(z_channels//4, 7, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(7, 49, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(49, 147, kernel_size=3, stride=1, padding=1),

        )
    def forward(self,z):
        # print(z.shape)
        z = z.reshape((-1, self.z_channels//4, 32, 32))
        x = self.decoder(z)
        x = x.reshape(-1, self.ch, 224, 224)
        # print(x.shape)
        # print('============================')
        return x











