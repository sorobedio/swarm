import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from utils.util import  instantiate_from_config
from stage2.set_transformer.models import SetTransformer
from stage2.set_transformer.super_linear import *



class EmbedData(nn.Module):
    def __init__(self,  enconfig, deconfig, **kwargs):
        super(EmbedData, self).__init__()
        self.intra = SetTransformer(enconfig, deconfig)
        self.inter = SetTransformer(enconfig, deconfig)

        self.fc_out = nn.Linear(512, 1024)

    def forward(self, inputs, *args, **kwargs):
        outputs = []
        print('============================')
        if isinstance(inputs, dict):

            inputs = [inputs]
        if isinstance(inputs, list):

            for inputs_lists in inputs:
                # print('=============================')
                # inputs_lists = list(x.values())

                # flat_list = [item for sublist in inputs_lists for item in sublist]

                bc =[]
                print(inputs_lists[0].shape)
                print('---------------')
                for bs in inputs_lists:
                    # print(bs)
                    bs = bs.cuda()
                    bs = bs.unsqueeze(0)
                    sc = self.intra(bs).squeeze(1)
                    bc.append(sc)

                bc = torch.cat(bc, dim=0)

                bc = bc.unsqueeze(0)
                out = self.inter(bc).reshape(-1)
                outputs.append(out)

        outputs = torch.stack(outputs, 0)
        outputs = self.fc_out(outputs)
        outputs = outputs.reshape(-1, 1, 32, 32)
        return outputs

class MIdentityCondStage(torch.nn.Module):
    def __init__(self, in_channels, input_size, **kwargs):
        super().__init__()
        self.in_channels =in_channels
        self.input_size = input_size

    def forward(self, x, *args, **kwargs):
        # print(len(x))
        x = x.reshape((-1, self.in_channels, self.input_size, self.input_size))
        return x


class IdentityCondStage(torch.nn.Module):
    def __init__(self, input_size, **kwargs):
        super().__init__()
        self.input_size = input_size

    def forward(self, x, *args, **kwargs):
        # x = x.reshape((-1, 5, 32, 32))
        x = x.to(torch.long)
        return x



# class MLPEncoder(nn.Module):
#     def __init__(self, in_dim=1000, out_dim=2304, **kwargs):
#         super(MLPEncoder, self).__init__()
#         self.out_dim = out_dim
#
#         self.in_dim=in_dim
#         self.dense1 = nn.Linear(self.in_dim, self.out_dim)
#
#     def forward(self, x):
#
#         x = self.dense1(x)
#
#         return x


class MyMLPEncoder(nn.Module):
    def __init__(self, in_ch=1, num_samples=5, input_size=32, num_classes=10, out_dim=4, embed_dim=512, **kwargs):
        super(MyMLPEncoder, self).__init__()
        self.in_ch = in_ch
        self.num_sample=num_samples
        self.n_classes = num_classes
        self.max_classes = num_classes
        self.max_dim = num_samples*num_classes
        self.in_res = input_size
        self.out_dim = out_dim
        self.embed_dim= embed_dim
        infeat = embed_dim*out_dim
        self.dense1 = LinearSuper(super_in_dim=num_samples*num_classes, super_out_dim=out_dim)
        self.dense2 = nn.Linear(infeat, in_ch*input_size*input_size)
        # self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        # print(len(inputs))
        # print('------------------------')
        out =[]
        for x in inputs:
            # print(len(x))
            if isinstance(x, list):
                if len(x)==1:
                    x= x[0]
                # x = torch.stack(x, dim=0)
            assert len(x.shape)==3, "x  should have 3 dimensions"
            ns = x.shape[1]
            nc = x.shape[0]
            dim = nc*ns
            if dim > self.max_dim :
                dim = self.max_dim
                x = x[:self.max_classes]
            # else:
            self.dense1.set_sample_config(dim, self.out_dim)

            x = x.cuda()
            # print(x.shape)

            x = rearrange(x, 'c n d -> d (n c)')
            # print(x.shape)
            x = self.dense1(x)
            x = F.leaky_relu(x)
            out.append(x)
        out = torch.stack(out, 0)
        x = rearrange(out, 'b d n -> b (d n)')
        x = self.dense2(x)
        # x = rearrange(x, 'b d n -> b (d n)')
        x = x.reshape(-1, self.in_ch, self.in_res, self.in_res)
        # print(x.shape)
        return x




class MLPEncoder(nn.Module):
    def __init__(self, in_ch=1, num_sample=5, in_res=32, n_classes=10, **kwargs):
        super(MLPEncoder, self).__init__()
        self.in_ch = in_ch
        self.num_sample=num_sample
        self.n_classes = n_classes
        self.in_res = in_res
        self.dense1 = nn.Linear(num_sample*n_classes, 10)
        self.dense2 = nn.Linear(5120, in_ch*in_res*in_res)
        # self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.stack(x, dim=0)
        x = x.cuda()
        x = rearrange(x, 'b c n d -> b d (n c)')
        # print(x.shape)
        x = self.dense1(x)
        x = F.leaky_relu(x)
        x = rearrange(x, 'b d n -> b (d n)')
        x = self.dense2(x)
        x = x.reshape(-1, self.in_ch, self.in_res, self.in_res)
        return x


class MiEmbedData(nn.Module):
    def __init__(self,  enconfig, deconfig, input_size=64, channels=1,  **kwargs):
        super(MiEmbedData, self).__init__()
        self.intra = SetTransformer(enconfig, deconfig)
        self.inter = SetTransformer(enconfig, deconfig)
        self.input_size= input_size
        self.channels = channels
        self.fc_out = nn.Linear(512, channels*input_size*input_size)

    def forward(self, inputs):
        outputs = []
        # print(len(inputs))
        for x in inputs:
            x = x.cuda()
            z = self.intra(x).squeeze(1)
            z = z.unsqueeze(0)
            out = self.inter(z).reshape(-1)
            outputs.append(out)
        outputs = torch.stack(outputs, 0)
        # print(outputs.shape)
        outputs = self.fc_out(outputs)
        outputs = outputs.reshape(-1, self.channels, self.input_size, self.input_size)
        return outputs


class KEmbedData(nn.Module):
    def __init__(self,  enconfig, deconfig, input_size=32, channels=1,  **kwargs):
        super(KEmbedData, self).__init__()
        self.intra = SetTransformer(enconfig, deconfig)
        self.inter = SetTransformer(enconfig, deconfig)
        self.input_size= input_size
        self.channels = channels
        self.proj = nn.Linear(512, channels*input_size*input_size)

    def forward(self, inputs):
        # print(inputs.shape)
        # inputs = inputs.reshape(-1, 10, 1, 512)

        outputs = []
        # print(len(inputs))
        for x in inputs:
            # print(x.shape)
            if isinstance(x, list):
                if isinstance(x[0], list):
                    if len(x[0])>1:
                        y = torch.stack(x[0], 0)
                    else:
                        y = x[0]
                if len(x) > 1:
                    y= torch.stack(x, 0)
                else:
                    y = x[0]
                    # y =x[0]
            else:
                if x.shape[0]==1:
                    x= x[0]
                y =x
            # print(y.shape)
            y = y.cuda()
            z = self.intra(y).squeeze(1)
            z = z.unsqueeze(0)
            out = self.inter(z).reshape(-1)
            outputs.append(out)


        outputs = torch.stack(outputs, 0)
        # print(outputs.shape)
        outputs = self.proj(outputs)
        outputs = outputs.reshape(-1, self.channels, self.input_size, self.input_size)
        return outputs


class MyEmbedData(nn.Module):
    def __init__(self,  enconfig, deconfig, input_size=48, channels=1,  **kwargs):
        super(MyEmbedData, self).__init__()
        self.intra = SetTransformer(enconfig, deconfig)
        self.inter = SetTransformer(enconfig, deconfig)
        self.input_size= input_size
        self.channels = channels
        self.fc_out = nn.Linear(512, 2304)

    def forward(self, inputs):
        # print(inputs.shape)
        # inputs = inputs.reshape(-1, 10, 1, 512)

        outputs = []
        # print(len(inputs))
        for x in inputs:
            if isinstance(x[0], list):
                y = torch.stack(x[0], 0)
            else:
                y = torch.stack(x, 0)
            y = y.cuda()
            z = self.intra(y).squeeze(1)
            z = z.unsqueeze(0)
            out = self.inter(z).reshape(-1)
            outputs.append(out)


        outputs = torch.stack(outputs, 0)
        # print(outputs.shape)
        outputs = self.fc_out(outputs)
        outputs = outputs.reshape(-1, 1, self.input_size, self.input_size)
        return outputs

class QueryEncoder(torch.nn.Module):
    def __init__(self, args):
        super(QueryEncoder, self).__init__()
        self.args = args
        self.fc = torch.nn.Linear(512, self.args.n_dims)

    def forward(self, D):
        q = []
        for d in D:
            _q = self.fc(d)
            _q = torch.mean(_q, 0)
            _q = self.l2norm(_q.unsqueeze(0))
            q.append(_q)
        q = torch.stack(q).squeeze()
        return q

    def l2norm(self, x):
        norm2 = torch.norm(x, 2, dim=1, keepdim=True)
        x = torch.div(x, norm2)
        return x