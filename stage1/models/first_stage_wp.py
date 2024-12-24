import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from stage1.modules.modules import Encoder, Decoder
# from stage1.modules.lemodule import Encoder, Decoder
# from modules.vqvae_module import VectorQuantizer, VectorQuantizerEMA, VqDecoder, VqEncoder
from stage1.modules.distributions import DiagonalGaussianDistribution

from utils.util import instantiate_from_config

class AutoencoderKL(nn.Module):
    def __init__(self, ddconfig,
                 lossconfig,
                 embed_dim,
                 learning_rate,
                 pred_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 input_key="weight",
                 cond_key="weight",
                 devive ='cuda',):
        super(AutoencoderKL, self).__init__()
        self.learning_rate = learning_rate
        self.ckpt_path= ckpt_path
        self.ignore_keys = ignore_keys
        self.input_key = input_key
        self.cond_key = cond_key
        self.embed_dim = embed_dim
        self.device=devive

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        if pred_config is not None:
            self.loss.disc = instantiate_from_config(pred_config)
            self.loss.set_predictor_grad(False)

        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        # self.encode_layer = nn.Sequential(nn.Flatten(),
        #                                   nn.Linear(2048, 1024),
        #                                   nn.LeakyReLU(),
        #                                   nn.Linear(1024, 2048),
        #                                   nn.LeakyReLU()
        #                                   )
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
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
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        # z = self.encode_layer(z)
        # z = z.reshape((-1, 32, 8, 8))
        # print(z.shape)
        # exit()
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec



    def forward(self, batch, sample_posterior=True):
        if isinstance(batch, dict):
            inputs = self.get_input(batch, self.input_key)
        else:
            inputs = batch
        posterior = self.encode(inputs)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        # print(z.shape)
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        return x
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  # list(self.encode_layer.parameters()),
                                  lr=lr, betas=(0.5, 0.9))

        # opt_ae = torch.optim.SGD(list(self.encoder.parameters()) +
        #                           list(self.decoder.parameters()) +
        #                           list(self.quant_conv.parameters()) +
        #                           list(self.post_quant_conv.parameters()),
        #                           lr=lr, momentum=0.9, weight_decay=0.0, nesterov=True)

        return opt_ae

    def get_last_layer(self):
        return self.decoder.conv_out.weight