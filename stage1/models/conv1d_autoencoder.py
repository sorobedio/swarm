import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from stage1.modules.conv1_encoding import Encoder, Decoder
from stage1.modules.conv1d_distributions import DiagonalGaussianDistribution
from utils.util import instantiate_from_config
from stage1.modules.losses.CustomLosses import ChunkWiseReconLoss


 # Define Log-Cosh Loss
def log_cosh_loss(y_pred, y_true):
    return torch.mean(torch.log(torch.cosh(y_pred - y_true)))

class Autoencoder(nn.Module):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 learning_rate,
                 ckpt_path=None,
                 ignore_keys=[],
                 input_key="weight",
                 cond_key="dataset",
                 device='cuda',
                 monitor=None,
                 ):
        super().__init__()
        self.devices =device
        self.cond_key = cond_key
        self.learning_rate =  learning_rate
        self.input_key = input_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv1d(2*ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv1d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        # print(x.dtype)
        h = self.encoder(x)
        z= self.quant_conv(h)

        return z

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        if isinstance(input, dict):
            input = input[self.input_key].to(self.devices)
        z = self.encode(input)

        dec = self.decode(z)
        dec = dec.reshape(input.shape)
        return input, dec

    def get_input(self, batch, k):
        x = batch[k].to(self.devices)
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs, reconstructions = self(batch)
        # reconstructions
        # loss = log_cosh_loss(reconstructions, inputs)*1000.0
        loss = F.mse_loss(inputs, reconstructions) * 1000

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.input_key)
        inputs, reconstructions = self(batch)
        # reconstructions
        # loss = log_cosh_loss(reconstructions, inputs)*1000.0
        loss = F.mse_loss(inputs, reconstructions) * 1000

        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight




class AENoDiscModel(Autoencoder):
    def __init__(self,ddconfig,
                 embed_dim,
                 learning_rate,
                 ckpt_path=None,
                 ignore_keys=[],
                 input_key="weight",
                 cond_key="dataset",
                 device='cuda',
                 monitor=None,
                 ):
        super().__init__(ddconfig=ddconfig, embed_dim=embed_dim,
                         ckpt_path=ckpt_path, ignore_keys=ignore_keys, input_key=input_key,
                         cond_key= cond_key, learning_rate=learning_rate)
        self.devices = device

    def training_step(self, batch, batch_idx):

        inputs, reconstructions= self(batch)
        # reconstructions
        # loss = log_cosh_loss(reconstructions, inputs)*1000.0
        loss = F.mse_loss(inputs, reconstructions)*1000

        return loss

    def validation_step(self, batch, batch_idx):

        inputs, reconstructions = self(batch)
        # loss = log_cosh_loss(reconstructions, inputs) * 1000.0
        loss = F.mse_loss(inputs, reconstructions)*1000.0

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=self.learning_rate, betas=(0.5, 0.9))
        return optimizer





class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x

