import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from stage1.modules.conv1_encoding import Encoder, Decoder
from stage1.modules.conv1d_distributions import DiagonalGaussianDistribution
from utils.util import instantiate_from_config
from stage1.modules.losses.CustomLosses import ChunkWiseReconLoss
import torch.distributions as dist


# Define Log-Cosh Loss
def log_cosh_loss(y_pred, y_true):
    return torch.mean(torch.log(torch.cosh(y_pred - y_true)))


# Define Student's T KL Divergence Loss
def student_t_kl_loss(mu, logvar, df):
    prior = dist.StudentT(df, torch.zeros_like(mu), torch.ones_like(mu))
    posterior = dist.StudentT(df, mu, torch.exp(0.5 * logvar))
    return torch.mean(torch.distributions.kl_divergence(posterior, prior))


class TVAE(nn.Module):
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
                 lambda_recon=1.0,
                 lambda_kl=0.1):
        super().__init__()
        self.devices = device
        self.cond_key = cond_key
        self.learning_rate = learning_rate
        self.input_key = input_key
        self.lambda_recon = lambda_recon
        self.lambda_kl = lambda_kl
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv1d(2 * ddconfig["z_channels"], embed_dim, 1)
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
        for k in list(sd.keys()):
            if any(k.startswith(ik) for ik in ignore_keys):
                del sd[k]
        self.load_state_dict(sd, strict=False)

    def encode(self, x):
        h = self.encoder(x)
        z_params = self.quant_conv(h)
        mu, logvar, logdf = torch.chunk(z_params, 3, dim=1)
        df = torch.exp(logdf) + 2.0  # Ensure df > 2 for stability
        eps = dist.StudentT(df).rsample(mu.shape)  # Sample from Student's T
        z = mu + torch.exp(0.5 * logvar) * eps  # Reparameterization trick
        return z, mu, logvar, df

    def decode(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, input):
        if isinstance(input, dict):
            input = input[self.input_key].to(self.devices)
        z, mu, logvar, df = self.encode(input)
        dec = self.decode(z)
        return input, dec, mu, logvar, df

    def get_input(self, batch, k):
        return batch[k].to(self.devices)

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs, reconstructions, mu, logvar, df = self(batch)
        recon_loss = log_cosh_loss(reconstructions, inputs)
        kl_loss = student_t_kl_loss(mu, logvar, df)
        loss = self.lambda_recon * recon_loss + self.lambda_kl * kl_loss
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.input_key)
        inputs, reconstructions, mu, logvar, df = self(batch)
        recon_loss = log_cosh_loss(reconstructions, inputs)
        kl_loss = student_t_kl_loss(mu, logvar, df)
        loss = self.lambda_recon * recon_loss + self.lambda_kl * kl_loss
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_vae = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.post_quant_conv.parameters()),
            lr=lr, betas=(0.5, 0.9))
        return [opt_vae], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight
