import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stage1.modules.modules import Encoder, Decoder
from stage1.modules.distributions import DiagonalGaussianDistribution
from utils.util import instantiate_from_config
from stage1.modules.losses.CustomLosses import ChunkWiseReconLoss
import torch.distributions as dist


# Define Log-Cosh Loss
def log_cosh_loss(y_pred, y_true):
    return torch.mean(torch.log(torch.cosh(y_pred - y_true)))


class TAutoencoder(nn.Module):
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
                 lambda_recon=1.0):
        super().__init__()
        self.devices = device
        self.cond_key = cond_key
        self.learning_rate = learning_rate
        self.input_key = input_key
        self.lambda_recon = lambda_recon
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 3 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
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
        df = torch.clamp(torch.exp(logdf) + 2.0, min=2.1, max=50.0)  # Ensure df is stable
        eps = dist.StudentT(df).rsample(mu.shape)  # Sample from Student's T
        z = mu + torch.exp(0.5 * logvar) * eps  # Reparameterization trick
        return z

    def decode(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, input):
        if isinstance(input, dict):
            input = input.get(self.input_key, None)
            if input is None:
                raise ValueError(f"Missing key {self.input_key} in input dictionary.")
            input = input.to(self.devices)
        z = self.encode(input)
        dec = self.decode(z)
        return input, dec

    def get_input(self, batch, k):
        return batch[k].to(self.devices)

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.input_key)
        inputs, reconstructions = self(inputs)
        recon_loss = log_cosh_loss(reconstructions, inputs)
        loss = self.lambda_recon * recon_loss
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.input_key)
        inputs, reconstructions = self(inputs)
        recon_loss = log_cosh_loss(reconstructions, inputs)
        loss = self.lambda_recon * recon_loss
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.post_quant_conv.parameters()),
            lr=lr, betas=(0.5, 0.9))
        return [opt_ae], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight


class AENoDiscModel(Autoencoder):
    def __init__(self,ddconfig,
                 embed_dim,
                 learning_rate,
                 beta_scheduler_config=None,
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
        self.is_kl_beta = False
        self.gl_step=0
        # if beta_scheduler_config is not None:
        #     self.is_kl_beta = True
        #     self.beta_scheduler = instantiate_from_config(beta_scheduler_config)  # annealing of temp

    # def beta_scheduling(self, global_step):
    #     self.loss.kl_weight = self.beta_scheduler.get_beta(global_step)

    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.input_key)
        inputs, reconstructions, z, df = self(inputs)
        recon_loss  = F.mse_loss(reconstructions, inputs, reduction="mean") * 1000.0
        # recon_loss = log_cosh_loss(reconstructions, inputs)
        latent_loss = student_t_kl_loss(z, df)
        loss = self.lambda_recon * recon_loss + self.lambda_latent * latent_loss



        return loss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.input_key)
        inputs, reconstructions, z, df = self(inputs)
        recon_loss = F.mse_loss(reconstructions, inputs, reduction="mean") * 1000.0
        # recon_loss = log_cosh_loss(reconstructions, inputs)
        latent_loss = student_t_kl_loss(z, df)
        loss = self.lambda_recon * recon_loss + self.lambda_latent * latent_loss
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  # list(self.loss.parameters()),
                                  lr=self.learning_rate, betas=(0.5, 0.95), weight_decay=4e-5)
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
