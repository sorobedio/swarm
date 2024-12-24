
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_dim, z_channels, input_size=64, in_channels=24, my_channels=49,
                 double_z=True, fdim=16384, tanh_out=False):
        super(Encoder, self).__init__()
        self.in_dim = in_dim
        self.fdim = fdim
        self.my_channels = my_channels
        self.tanh_out =tanh_out

        if double_z:
            self.z_channels = z_channels*2
        else:
            self.z_channels = z_channels
        self.input_size = input_size
        self.in_channels= in_channels
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv = nn.Conv2d(128, self.z_channels, kernel_size=3, stride=2, padding=1, bias=True)

        self.d1 = nn.Linear(in_dim, self.fdim)

    def forward(self, x):
        # print(x.shape)
        x = x.reshape(-1, self.my_channels, self.in_dim)  # bx256*256*3-->bx24x64*64*2
        x = self.d1(x)
        x = F.leaky_relu(x)
        x = x.reshape(-1, self.in_channels, self.input_size, self.input_size)
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        # z = self.conv3(x)
        x = F.leaky_relu(x)
        z = self.conv(x)
        # z = F.leaky_relu(z)
        return z



class Decoder(nn.Module):
    def __init__(self, in_dim, z_channels, input_size=64, in_channels=24, my_channels=49,
                 double_z=True, fdim=16384, tanh_out=False):
        super(Decoder, self).__init__()
        self.in_dim = in_dim

        self.ldim= input_size*input_size
        self.out_channels = in_channels
        self.in_channels = in_channels
        self.my_channels= my_channels
        self.tanh_out=tanh_out
        self.fdim = fdim
        if double_z:
            self.z_channels = z_channels*2
        else:
            self.z_channels = z_channels

        self.d1 = nn.Linear(self.fdim, in_dim)
        self.up1 = nn.ConvTranspose2d(self.z_channels, 128, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        # self.up4 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Conv2d(64, self.in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        z = self.up1(z)
        z = F.leaky_relu(z)
        z = self.up2(z)
        z = F.leaky_relu(z)
        z = self.up3(z)
        # z = F.leaky_relu(z)
        # z = self.up4(z)
        z = F.leaky_relu(z)
        z = self.conv(z)

        z = z.reshape(-1, self.my_channels, self.fdim)
        z = self.d1(z)
        if self.tanh_out:
            # z = torch.tanh(z)
            z = F.leaky_relu(z)
        # z = torch.tanh(z)
        # z = torch.sigmoid(z)
        x = z.reshape(-1, self.my_channels * self.in_dim)
        return x





class MixEncoder(nn.Module):
    def __init__(self, in_dim, z_channels, input_size=64, in_channels=24, my_channels=49,  double_z=True, fdim=16384):
        super(MixEncoder, self).__init__()
        self.in_dim = in_dim
        self.fdim = fdim
        self.my_channels = my_channels

        if double_z:
            self.z_channels = z_channels*2
        else:
            self.z_channels = z_channels
        self.input_size = input_size
        self.in_channels= in_channels
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv = nn.Conv2d(128, self.z_channels, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv1d =  nn.Conv1d(4096, in_channels, kernel_size=3, stride=2, padding=1)

        # self.d1 = nn.Linear(in_dim, self.fdim)

    def forward(self, x):
        # print(x.shape)
        x = x.reshape(-1, self.my_channels, self.in_dim)  # bx256*256*3-->bx24x64*64*2
        x = self.conv1d(x)
        x = F.leaky_relu(x)
        x = x.reshape(-1, self.in_channels, self.input_size, self.input_size)
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        # z = self.conv3(x)
        x = F.leaky_relu(x)
        z = self.conv(x)
        # z = F.leaky_relu(z)
        return z



class MixDecoder(nn.Module):
    def __init__(self, in_dim, z_channels, input_size=64, in_channels=24, my_channels=49,  double_z=True, fdim=16384):
        super(MixDecoder, self).__init__()
        self.in_dim = in_dim

        self.ldim= input_size*input_size
        self.out_channels = in_channels
        self.in_channels = in_channels
        self.my_channels= my_channels
        self.fdim = fdim
        if double_z:
            self.z_channels = z_channels*2
        else:
            self.z_channels = z_channels

        # self.d1 = nn.Linear(self.fdim, in_dim)
        self.up1 = nn.ConvTranspose2d(self.z_channels, 128, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        # self.up4 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Conv2d(64, self.in_channels, kernel_size=3, stride=1, padding=1)
        self.up1d = nn.ConvTranspose1d(in_channels, my_channels, kernel_size=4, stride=2, padding=1)
        self.conv1d = nn.Conv1d(my_channels, my_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        z = self.up1(z)
        z = F.leaky_relu(z)
        z = self.up2(z)
        z = F.leaky_relu(z)
        z = self.up3(z)
        # z = F.leaky_relu(z)
        # z = self.up4(z)
        z = F.leaky_relu(z)
        z = self.conv(z)
        z = z.reshape(-1, self.in_channels, self.fdim)
        z = self.up1d(z)
        z = self.conv1d(z)
        x = z.reshape(-1, self.my_channels, self.in_dim)
        return x



class MyEncoder(nn.Module):
    def __init__(self, in_dim, z_channels, input_size=64, in_channels=24, double_z=True):
        super(MyEncoder, self).__init__()
        self.in_dim = in_dim

        if double_z==True:
            self.z_channels = z_channels*2
        else:
            self.z_channels = z_channels
        self.input_size = input_size
        self.in_channels= in_channels
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(32, self.z_channels, kernel_size=3, stride=2, padding=1)
        # self.d1 = nn.Linear(in_dim, self.input_size*self.input_size)

    def forward(self, x):
        # x = x.reshape(-1, self.in_channels, self.in_dim)  # bx256*256*3-->bx24x64*64*2
        # x = self.d1(x)
        # x = F.leaky_relu(x)
        x = x.reshape(-1, self.in_channels, 256, 256)  # bx256*256*3-->bx24x64*64*2
        # x = x.reshape(-1, self.in_channels, self.input_size, self.input_size)
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = self.conv4(x)
        x = F.leaky_relu(x)
        z = self.conv5(x)
        # x = F.leaky_relu(x)
        return z



class MyDecoder(nn.Module):
    def __init__(self, in_dim, z_channels, input_size=64, in_channels=24, double_z=True):
        super(MyDecoder, self).__init__()
        self.in_dim = in_dim

        self.ldim= input_size*input_size
        self.out_channels = in_channels
        self.in_channels = in_channels
        # if double_z==True:
        #     self.z_channels = z_channels*2
        # else:
        self.z_channels = z_channels
        # self.d1 = nn.Linear(self.ldim, in_dim)
        self.up1 = nn.ConvTranspose2d(self.z_channels, 32, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.up4 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.up5 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Conv2d(32, self.in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        z = self.up1(z)
        z = F.leaky_relu(z)
        z = self.up2(z)
        z = F.leaky_relu(z)
        z = self.up3(z)
        z = F.leaky_relu(z)
        z = self.up4(z)
        z = F.leaky_relu(z)
        z = self.up5(z)
        z = F.leaky_relu(z)
        z = self.conv(z)
        # z = F.leaky_relu(z)
        # z = z.reshape(-1, self.in_channels, self.ldim)
        # z = self.d1(z)
        #         z = torch.tanh(z)
        z = torch.sigmoid(z)
        x = z.reshape(-1, self.in_channels * self.in_dim)
        return x





class LEncoder(nn.Module):
    def __init__(self, in_dim, z_features, in_channels=16, double_z=True, fdim=12288):
        super(LEncoder, self).__init__()
        self.in_dim = in_dim
        self.out_channels = in_channels
        self.in_channels = in_channels
        self.fdim = fdim
        if double_z:
            z_features = z_features * 2
        self.z_features = z_features
        self.fdim = fdim
        self.in_channels= in_channels
        self.l1 = nn.Linear(4096, 1024)
        self.l2 = nn.Linear(1024, z_features)
        # self.l3 = nn.Linear(1024, 512)
        self.d1 = nn.Linear(in_dim, 4096)

    def forward(self, x):
        x = x.reshape(-1, self.in_channels, self.in_dim)  # bx256*256*3-->bx24x64*64*2
        x = self.d1(x)
        x = F.leaky_relu(x)
        x = self.l1(x)
        x = F.leaky_relu(x)
        z = self.l2(x)
        z = z.reshape((-1, self.in_channels*self.z_features))
        return z



class LDecoder(nn.Module):
    def __init__(self, in_dim, z_features, in_channels=16, double_z=True, fdim=12288):
        super(LDecoder, self).__init__()
        self.in_dim = in_dim
        self.out_channels = in_channels
        self.in_channels = in_channels
        self.fdim = fdim
        if double_z:
            z_features = z_features * 2
        self.z_features = z_features
        self.fdim = fdim
        self.d1 = nn.Linear(4096, in_dim)
        self.up1 = nn.Linear(z_features, 1024)
        self.up2 = nn.Linear(1024, 4096)

    def forward(self, z):
        z = z.reshape((-1, self.in_channels*self.z_features))
        z = self.up1(z)
        z = F.leaky_relu(z)
        z = self.up2(z)
        z = F.leaky_relu(z)
        z = self.d1(z)
        x = z.reshape(-1, self.in_channels * self.in_dim)
        return x


