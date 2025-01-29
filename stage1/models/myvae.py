import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stage1.modules.modules import Encoder, Decoder
from stage1.modules.distributions import DiagonalGaussianDistribution
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
                 latent_shape=None,
                 monitor=None,
                 ):
        super().__init__()
        self.devices = device
        self.cond_key = cond_key
        self.learning_rate = learning_rate
        self.input_key = input_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.chunk_loss = ChunkWiseReconLoss(step_size=128)

        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def adaptive_mesh(self, sigma):
        """
        Moving Mesh Adaptation: Adjusts latent sampling dynamically per sample
        without batch-wide dependencies.
        """
        batch_size = sigma.shape[0]
        monitor_func = torch.zeros_like(sigma)  # Ensure same shape as sigma

        # Compute per-sample gradient correctly
        for i in range(batch_size):
            grad_i = torch.autograd.grad(sigma[i].sum(), sigma[i], retain_graph=True, create_graph=True)[0]
            monitor_func[i] = torch.abs(grad_i) + 1e-4  # Store per-sample gradients

        # Normalize per sample
        adaptive_factor = monitor_func / (monitor_func.mean(dim=(1, 2, 3), keepdim=True) + 1e-6)

        return sigma * adaptive_factor  # Warps latent distribution

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        posterior.std = self.adaptive_mesh(posterior.std)  # Apply adaptive mesh refinement
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
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
        dec = self.decode(z)
        dec = dec.reshape(input.shape)
        return input, dec, posterior

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.input_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            return aeloss

        if optimizer_idx == 1:
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
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
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
                 beta_scheduler_config=None,
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
        self.is_kl_beta = False
        self.gl_step=0
        if beta_scheduler_config is not None:
            self.is_kl_beta = True
            self.beta_scheduler = instantiate_from_config(beta_scheduler_config)  # annealing of temp

    def beta_scheduling(self, global_step):
        self.loss.kl_weight = self.beta_scheduler.get_beta(global_step)

    def training_step(self, batch, batch_idx):
        if self.is_kl_beta:
            self.beta_scheduling(self.gl_step)
        inputs, reconstructions, posterior = self(batch)
        # reconstructions
        mse = F.mse_loss(inputs, reconstructions)
        # cmse = self.chunk_loss(inputs, reconstructions)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior,  split="train")
        loss = aeloss# +mse*1000
        self.gl_step += 1
        # print(f"inputs: {inputs[0][:20]}")
        # print(f"reconstructions: {reconstructions[0][:20]}")
        # diff = inputs[0][:20] - reconstructions[0][:20]
        # print(f'diff: {diff}')
        # input_mean = inputs.mean().item()
        # recon_mean = reconstructions.mean().item()
        # print(f"Input Mean: {input_mean}, Reconstruction Mean: {recon_mean}")

        return loss, log_dict_ae

    def validation_step(self, batch, batch_idx):

        inputs, reconstructions, posterior = self(batch)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior,  split="val")
        # discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, self.global_step,
        #                                     last_layer=self.get_last_layer(), split="val")

        return aeloss

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

