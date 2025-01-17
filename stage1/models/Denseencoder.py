import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from stage1.modules.Linearmodules  import LEncoder, LDecoder
from stage1.modules.reparameterization import DiagonalGaussianDistribution
from utils.util import instantiate_from_config
from stage1.modules.losses.CustomLosses import ChunkWiseReconLoss

class AutoencoderKL(nn.Module):
    def __init__(self,
                 ddconfig,
                 lossconfig,
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
        self.encoder = LEncoder(**ddconfig)
        self.decoder = LDecoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.z_features=ddconfig["z_features"]
        # self.chunk_loss = ChunkWiseReconLoss(step_size=1024)
        assert ddconfig["double_z"]
        self.quant_fc = nn.Linear(2*ddconfig["z_features"]*ddconfig["in_channels"], 2*embed_dim)
        self.post_quant_fc = nn.Linear(embed_dim, 2 * ddconfig["z_features"]*ddconfig["in_channels"])
        # self.quant_fc = nn.Linear(2 * ddconfig["z_features"], 2 * embed_dim)
        # self.post_quant_fc = nn.Linear(embed_dim, 2 * ddconfig["z_features"])
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
        h = self.encoder(x)
        # print(h.size())
        # h = h.reshape(h.size(0), -1)
        moments = self.quant_fc(h)
        # print(moments.size())
        # exit()
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_fc(z)
        # b = z.size(0)
        # z = z.resize(b, self.z_features, -1)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        if isinstance(input, dict):
            input = input[self.input_key].to(self.devices)
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        print(z.shape)
        # exit()
        dec = self.decode(z)
        dec = dec.reshape(input.shape)
        return input, dec, posterior

    def get_input(self, batch, k):
        x = batch[k].to(self.devices)
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.input_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.input_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_fc.parameters())+
                                  list(self.post_quant_fc.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight




class VAENoDiscModel(AutoencoderKL):
    def __init__(self,ddconfig,
                 lossconfig,
                 embed_dim,
                 learning_rate,
                 ckpt_path=None,
                 ignore_keys=[],
                 input_key="weight",
                 cond_key="dataset",
                 device='cuda',
                 monitor=None,
                 ):
        super().__init__(ddconfig=ddconfig, lossconfig=lossconfig, embed_dim=embed_dim,
                         ckpt_path=ckpt_path, ignore_keys=ignore_keys, input_key=input_key,
                         cond_key= cond_key, learning_rate=learning_rate)
        self.devices = device

    def training_step(self, batch, batch_idx):

        inputs, reconstructions, posterior = self(batch)
        # reconstructions
        mse = F.mse_loss(inputs, reconstructions)
        # cmse = self.chunk_loss(inputs, reconstructions)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior,  split="train")
        loss = aeloss
        return loss, log_dict_ae

    def validation_step(self, batch, batch_idx):

        inputs, reconstructions, posterior = self(batch)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior,  split="val")
        # discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, self.global_step,
        #                                     last_layer=self.get_last_layer(), split="val")

        return aeloss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_fc.parameters())+
                                  list(self.post_quant_fc.parameters()),
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

