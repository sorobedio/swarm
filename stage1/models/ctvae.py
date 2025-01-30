import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stage1.modules.modules import Encoder, Decoder
from stage1.modules.distributions import DiagonalGaussianDistribution
from utils.util import instantiate_from_config
from stage1.modules.losses.CustomLosses import ChunkWiseReconLoss
import torch.distributions as dist


# Define Log-Cosh Loss - Made numerically stable
def log_cosh_loss(y_pred, y_true):
    diff = y_pred - y_true
    return torch.mean(torch.log(torch.cosh(diff + 1e-12)))


# Define Student's T KL Divergence Loss - Fixed stability issues
def student_t_kl_loss(mu, logvar, df):
    eps = 1e-6
    df = torch.clamp(df, min=2.1 + eps, max=50.0)
    var = torch.exp(logvar)

    # All tensors are already [batch_size, channels, height, width]
    # Just calculate components directly
    log_det = 0.5 * torch.sum(logvar, dim=[1, 2, 3])
    trace_term = 0.5 * torch.sum((mu ** 2 + var) / df, dim=[1, 2, 3])
    df_term = 0.5 * torch.sum(
        (df + 1) * torch.log1p(mu ** 2 / (df * var + eps)),
        dim=[1, 2, 3]
    )

    # Sum over spatial dimensions, mean over batch
    total = log_det + trace_term + df_term
    return torch.mean(total)


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
                 latent_shape=None,
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
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 3 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

        # Initialize weights for better convergence
        self.apply(self._init_weights)

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

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

        # Fixed numerical stability in distribution parameters
        df = torch.clamp(torch.exp(logdf) + 2.1, min=2.1, max=50.0)
        scale = torch.exp(0.5 * logvar)

        # Proper reparameterization for Student's T
        eps = torch.randn_like(mu)
        v = torch.empty_like(df).exponential_() * (1 / df)
        z = mu + scale * eps * torch.sqrt(df / v)
        # print(z.shape,mu.shape,df.shape, logdf.shape)
        # exit()

        return z, mu, logvar, df

    def decode(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, input):
        if isinstance(input, dict):
            input = input.get(self.input_key, None)
            if input is None:
                raise ValueError(f"Missing key {self.input_key} in input dictionary.")
            input = input.to(self.devices)
        z, mu, logvar, df = self.encode(input)
        dec = self.decode(z)
        return input, dec, mu, logvar, df

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        return x.to(self.devices)

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.input_key)
        inputs, reconstructions, mu, logvar, df = self(inputs)

        # Using the fixed loss functions
        recon_loss = log_cosh_loss(reconstructions, inputs)
        kl_loss = student_t_kl_loss(mu, logvar, df)
        loss = self.lambda_recon * recon_loss + self.lambda_kl * kl_loss

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.input_key)
        inputs, reconstructions, mu, logvar, df = self(inputs)
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




class AENoDiscModel(TVAE):
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
        logs ={}
        inputs = self.get_input(batch, self.input_key)
        inputs, reconstructions, mu, logvar, df = self(inputs)
        # recon_loss = log_cosh_loss(reconstructions, inputs)
        recon_loss = F.mse_loss(reconstructions, inputs, reduction="mean")
        # kl_loss = student_t_kl_loss(mu, logvar, df)
        # 2) KL divergence (Monte Carlo)
        print('computing_kl')
        kl_loss = student_t_kl_loss(mu, logvar, df)

        loss = self.lambda_recon * recon_loss + self.lambda_kl * kl_loss
        logs['recon_loss'] = recon_loss
        logs['kl_loss'] = kl_loss
        logs['loss'] = loss

        return loss, logs

    def validation_step(self, batch, batch_idx):
        logs = {}
        inputs = self.get_input(batch, self.input_key)
        inputs, reconstructions, mu, logvar, df = self(inputs)
        # recon_loss = log_cosh_loss(reconstructions, inputs)
        recon_loss = F.mse_loss(reconstructions, inputs, reduction="mean")
        # kl_loss = student_t_kl_loss(mu, logvar, df)
        # 2) KL divergence (Monte Carlo)
        kl_loss = student_t_kl_loss(mu, logvar, df)

        loss = self.lambda_recon * recon_loss + self.lambda_kl * kl_loss
        logs['recon_loss'] = recon_loss
        logs['kl_loss'] = kl_loss
        logs['loss'] = loss

        return loss, logs

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
