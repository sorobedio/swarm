
import os
import  numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def Normalize(in_channels, num_groups=2):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x



class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels,
                                    in_channels,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1)
    def forward(self, x):
        x = self.conv(x)
        return x


# class ResBlock(nn.Module):
#     def __init__(self, res_channel, dilation, n_mels, cond=True):
#         super().__init__()
#         self.dilated_conv = nn.Conv1d(res_channel, 2 * res_channel, 3, \
#                                    padding=dilation, dilation=dilation)
#         self.diffstep_proj = nn.Linear(512, res_channel)
#         self.cond_proj = nn.Conv1d(n_mels, 2 * res_channel, 1)
#         self.output_proj = nn.Conv1d(res_channel, 2 * res_channel, 1)
#         self.cond = cond
#
#     def forward(self, inp, diff_step, conditioner):
#         diff_step = self.diffstep_proj(diff_step).unsqueeze(-1)
#         x = inp + diff_step
#
#         # control whether to add condition
#         if self.cond:
#             conditioner = self.cond_proj(conditioner)
#             x = self.dilated_conv(x) + conditioner
#         gate, val = torch.chunk(x, 2, dim=1)  # gate function
#         x = torch.sigmoid(gate) * torch.tanh(val)
#
#         x = self.output_proj(x)
#         residual, skip = torch.chunk(x, 2, dim=1)
#         return (inp + residual) / np.sqrt(2.0), skip

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv1d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class SelfAttentionLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_Q = nn.Conv1d(in_channels, in_channels, kernel_size = 1, bias = False)
        self.conv_K = nn.Conv1d(in_channels, in_channels, kernel_size = 1, bias = False)
        self.conv_V = nn.Conv1d(in_channels, in_channels, kernel_size = 1, bias = False)
    def forward(self, x):
        Q = self.conv_Q(x)
        K = self.conv_K(x)
        V = self.conv_V(x)
        # A = Q.transpose(1, 2).matmul(K).softmax(2)
        # y = A.matmul(V.transpose(1, 2)).transpose(1, 2)
        A = torch.einsum('nct,ncs->nts', Q, K).softmax(2)
        y = torch.einsum('nts,ncs->nct', A, V)
        return y


class Encoder(nn.Module):
    def __init__(self, *, ch, in_dim, in_ch, z_channels, ch_mult=(1, 2, 4, 2), num_res_blocks,
                 attn_resolutions,  in_channels, resolution=32, double_z=True, fdim=8192,
                 dropout=0.0, **ignore_kwargs):
        super(Encoder, self).__init__()
        self.in_dim=in_dim
        self.in_ch = in_ch
        self.z_channels = z_channels
        self.ch = ch
        self.attn_resolution=attn_resolutions
        self.in_channels = in_channels,
        self.channels = in_channels
        # self.my_channels = my_channels
        self.fdim=fdim
        self.num_res_blocks=num_res_blocks
        self.resolution = resolution
        self.double_z=double_z
        n_down = len(ch_mult)-1
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)

        self.fc_in = nn.Linear(self.in_dim, fdim)
        self.conv_in = torch.nn.Conv1d(in_channels,
                                       ch ,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(SelfAttentionLayer(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = SelfAttentionLayer(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        2 * z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x):
        temb = None
        #adapt input shape
        x = x.reshape((-1, self.in_ch, self.in_dim))
        x = self.fc_in(x)
        # x = F.leaky_relu(x)
        x = x.reshape((-1, self.channels,  self.resolution))
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self,  *, ch, in_dim, in_ch, out_ch, z_channels, ch_mult=(1, 2, 4, 2), num_res_blocks,
                 attn_resolutions,  in_channels, resolution=32, give_pre_end=False, double_z=True, fdim=8192,
                 dropout=0.0, tanh_out=False, **ignore_kwargs):
        super(Decoder, self).__init__()
        self.in_dim = in_dim
        self.in_ch = in_ch
        self.z_channels = z_channels
        self.ch = in_channels
        self.out_ch=out_ch
        self.attn_resolution = attn_resolutions
        self.in_channels = in_channels,
        # self.my_channels = my_channels
        self.fdim = fdim
        self.fch =  in_ch
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.double_z = double_z
        n_down = len(ch_mult) - 1
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.dropout = dropout
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))
        self.fc_out = nn.Linear(self.fdim, self.in_dim)

        # z to block_in
        self.conv_in = torch.nn.Conv1d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = SelfAttentionLayer(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(SelfAttentionLayer(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        h = h.reshape((-1, self.in_ch, self.fdim))
        h = self.fc_out(h)
        h = h.reshape((-1, self.out_ch, 128))
        if self.tanh_out:
            h = torch.tanh(h)

        return h






class MyEncoder(nn.Module):
    def __init__(self, in_dim, z_channels, input_size=64, in_channels=24, my_channels=49,  double_z=True, fdim=16384):
        super(MyEncoder, self).__init__()
        self.in_dim = in_dim
        self.fdim = fdim
        self.my_channels = my_channels

        if double_z:
            self.z_channels = z_channels*2
        else:
            self.z_channels = z_channels
        self.input_size = input_size
        self.in_channels= in_channels
        self.conv1 = nn.Conv1d(self.in_channels, 64, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1, bias=True)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv = nn.Conv1d(128, self.z_channels, kernel_size=3, stride=2, padding=1, bias=True)

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



class MyDecoder(nn.Module):
    def __init__(self, in_dim, z_channels, input_size=64, in_channels=24, my_channels=49,  double_z=True, fdim=16384):
        super(MyDecoder, self).__init__()
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
        # z = torch.tanh(z)
        # z = torch.sigmoid(z)
        x = z.reshape(-1, self.my_channels * self.in_dim)
        return x

