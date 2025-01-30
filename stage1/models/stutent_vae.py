import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stage1.modules.modules import Encoder, Decoder
from stage1.modules.distributions import DiagonalGaussianDistribution
from utils.util import instantiate_from_config
from stage1.modules.losses.CustomLosses import ChunkWiseReconLoss
import torch.distributions as dist

# ------------------------------------------------------
# 1) Log-Cosh Reconstruction Loss
# ------------------------------------------------------
def log_cosh_loss(y_pred, y_true):
    return torch.mean(torch.log(torch.cosh(y_pred - y_true)))

# ------------------------------------------------------
# 2) Monte Carlo KL for StudentT
#    KL[q||p] = E_q[ log q(z) - log p(z) ]
# ------------------------------------------------------
def student_t_kl_loss(mu, logvar, df, n_samples=5):
    """
    Approximate KL[ StudentT(df, mu, scale_q) || StudentT(df, 0, 1) ]
    via Monte Carlo sampling.

    Args:
        mu:        Mean of the posterior StudentT ([B, C, H, W] or [B, latent_dim])
        logvar:    Log-variance of the posterior StudentT
        df:        Degrees of freedom of the posterior StudentT
        n_samples: How many samples to draw for MC approximation

    Returns:
        A scalar (float) approximate KL divergence
    """
    # Posterior distribution q(z)
    scale_q = torch.exp(0.5 * logvar)
    q = dist.StudentT(df, loc=mu, scale=scale_q)

    # Prior distribution p(z): same df, zero mean, unit scale
    scale_p = torch.ones_like(mu)
    p = dist.StudentT(df, loc=torch.zeros_like(mu), scale=scale_p)

    # Draw samples from q: shape => [n_samples, B, ...]
    z_samples = q.rsample((n_samples,))

    # Compute log probabilities under q and p
    log_qz = q.log_prob(z_samples)  # shape [n_samples, B, ...]
    log_pz = p.log_prob(z_samples)  # same shape

    # KL = E_q[log q(z) - log p(z)]
    kl_per_sample = log_qz - log_pz  # [n_samples, B, ...]
    kl_mean_over_samples = kl_per_sample.mean(dim=0)  # [B, ...]
    kl = kl_mean_over_samples.mean()  # scalar
    return kl

# ------------------------------------------------------
# 3) TVAE Model
# ------------------------------------------------------
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

        # ddconfig["double_z"] typically means we produce mu+logvar in channels
        # Here we produce mu, logvar, and log-df in channels => 3 * embed_dim
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
        # 1) Encode input to latent feature map
        h = self.encoder(x)
        # 2) Project to (mu, logvar, logdf)
        z_params = self.quant_conv(h)
        mu, logvar, logdf = torch.chunk(z_params, 3, dim=1)

        # 3) Degrees of freedom: clamp to [2.1, 50]
        df = torch.clamp(torch.exp(logdf) + 2.1, min=2.1, max=50.0)

        # 4) Sample from Student's T using reparameterization
        #    z = mu + scale * eps, where eps ~ StudentT(df, 0, 1).
        eps = dist.StudentT(df).rsample(mu.shape)
        z = mu + torch.exp(0.5 * logvar) * eps
        # print('encode x')
        return z, mu, logvar, df

    def decode(self, z):
        # print('decoding z')
        z = self.post_quant_conv(z)
        x =self.decoder(z)
        return x

    def forward(self, input):
        # If we get a dict, we fetch the relevant key
        if isinstance(input, dict):
            input = input.get(self.input_key, None)
            if input is None:
                raise ValueError(f"Missing key {self.input_key} in input dictionary.")
            input = input.to(self.devices)

        # Encode -> sample -> decode
        z, mu, logvar, df = self.encode(input)
        dec = self.decode(z)
        return input, dec, mu, logvar, df

    def get_input(self, batch, k):
        return batch[k].to(self.devices)

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.input_key)
        inputs, reconstructions, mu, logvar, df = self(inputs)

        # 1) Reconstruction loss
        recon_loss = log_cosh_loss(reconstructions, inputs)

        # 2) KL divergence (Monte Carlo)
        kl_loss = student_t_kl_loss(mu, logvar, df, n_samples=5)

        # Weighted sum
        loss = self.lambda_recon * recon_loss + self.lambda_kl * kl_loss
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.input_key)
        inputs, reconstructions, mu, logvar, df = self(inputs)

        recon_loss = log_cosh_loss(reconstructions, inputs)
        kl_loss = student_t_kl_loss(mu, logvar, df, n_samples=5)
        loss = self.lambda_recon * recon_loss + self.lambda_kl * kl_loss
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_vae = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.post_quant_conv.parameters()),
            lr=lr, betas=(0.5, 0.9)
        )
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
        # print('computing_kl')
        kl_loss = student_t_kl_loss(mu, logvar, df, n_samples=5)

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
        kl_loss = student_t_kl_loss(mu, logvar, df, n_samples=5)

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
