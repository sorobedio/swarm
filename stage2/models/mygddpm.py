
import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from einops import rearrange, reduce, repeat
from torch import optim
from utils import *
from stage2.modules.tinymodules import UNet_conditional, EMA
import logging
from torch.utils.tensorboard import SummaryWriter
from utils.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from stage1.modules.distributions import normal_kl, DiagonalGaussianDistribution
from stage1.modules.losses.CustomLosses import ChunkWiseReconLoss


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class MyDiffusion(nn.Module):
    def __init__(self, unet_config,
                 first_stage_config,
                 cond_stage_config,
                 learning_rate,
                 noise_steps=1000,
                 beta_start=1e-4,
                 beta_end=0.02,
                 input_size=32,
                 latent_size=4,
                 latent_ch=64,
                 channels=1,
                 embdim=32,
                 num_timesteps_cond=None,
                 cond_stage_key="dataset",
                 first_stage_key='weight',
                 cond_stage_trainable=False,
                 concat_mode=True,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 loss_type='l1',
                 device="cuda",
                 *args, **kwargs):
        self.channels = channels
        self.loss_type=loss_type
        self.first_stage_key=first_stage_key
        self.learning_rate = learning_rate
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        self.latent_size = latent_size
        self.latent_ch = latent_ch
        self.embdiom = embdim
        self.scale_factor = scale_factor
        self.device = device
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super(MyDiffusion, self).__init__()
        self.loss_weight = 1.0

        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        # noise_steps = 1000
        self.num_timesteps = noise_steps

        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.learn_logvar = learn_logvar
        self.logvar = nn.Parameter(torch.ones(size=()))

        self.closs = ChunkWiseReconLoss(step_size=48)
        # self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        # if self.learn_logvar:
        #     self.logvar = nn.Parameter(torch.ones(size=()))
        #     self.logvar = nn.Parameter(self.logvar, requires_grad=True)


        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end


        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)



        self.input_size = input_size
        self.device = device


        self.model = DiffusionWrapper(unet_config, conditioning_key)
        self.optimizer = self.configure_optimizers()


        # self.model_ema = LitEma(self.model)
        # self.ema = EMA(0.995)
        # self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        # model = torch.load('aecheckpoints/best_valid_loss_lenetimg_.pth')
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        z = z.reshape(-1, self.channels, self.input_size, self.input_size)
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        c = self.cond_stage_model(c)
        return c

    @torch.no_grad()
    def get_input(self, batch):
        x = batch[self.first_stage_key]
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()
        z = z.reshape(-1, self.channels, self.input_size, self.input_size)

        cond_key = self.cond_stage_key
        xc = batch[cond_key]
        if isinstance(xc, torch.Tensor):
            c = self.get_learned_conditioning(xc.to(self.device))
        else:
            c = self.get_learned_conditioning(xc)
        # out = [z, c]

        return z, [c]

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1. / self.scale_factor * z
        z = z.reshape(-1, self.latent_ch, self.latent_size, self.latent_size)
        return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)


    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_weights(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, n=1, cond=None, cfg_scale=3):
        # print(len(cond))
        # if isinstance(cond, list):
        #     cond =
        if n> 1 and len(cond)==1:
            cond = n*cond
        cond = self.get_learned_conditioning(cond)
        # n = len(cond)
        logging.info(f"Sampling {n} new images....")
        self.model.eval()
        if not isinstance(cond, list):
            cond =[cond]
        # print(cond[0].shape)
        # if not isinstance(cond, list):
        #     cond =[cond]
        with torch.no_grad():
            x = torch.randn((n, self.channels, self.input_size, self.input_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = self.model(x, t, cond)
                # if cfg_scale > 0:
                #     uncond_predicted_noise = self.model(x, t, None)
                #     predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        self.model.train()
        # x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)
        x = self.decode_first_stage(x)
        return x

    def get_loss(self, pred, target, t=None, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
                loss = reduce(loss, 'b ... -> b', 'mean')

        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        # loss = torch.nn.functional.mse_loss(pred, target, reduction='none')
        # loss = loss * extract(self.loss_weight, t, loss.shape)
        #
        #
        # loss =  loss.mean()

        return loss*1000

    def update_ema(self):
        self.ema.step_ema(self.ema_model, self.model)

    def train_step(self, batch):
        x, cond = self.get_input(batch)
        self.optimizer.zero_grad()
        t = self.sample_timesteps(x.shape[0]).to(self.device)
        x_t, noise = self.noise_weights(x, t)

        predicted_noise = self.model(x_t, t, cond)
        myloss = self.closs(noise, predicted_noise)*100
        loss = self.get_loss(noise, predicted_noise, t=t) + myloss
        loss.backward()
        self.optimizer.step()
        return loss


    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        # if self.use_scheduler:
        #     assert 'target' in self.scheduler_config
        #     schedulers = instantiate_from_config(self.scheduler_config)
        #
        #     print("Setting up LambdaLR schedulers...")
        #     schedulers = [
        #         {
        #             'schedulers': LambdaLR(opt, lr_lambda=schedulers.schedule),
        #             'interval': 'step',
        #             'frequency': 1
        #         }]
        #     return opt, schedulers
        return opt


class DiffusionWrapper(nn.Module):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key

        # self.input_adapt = nn.Sequential(nn.Linear(2304, 4096))

        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        # x = torch.flatten(x, start_dim=1)
        # x = self.input_adapt(x)
        # x = x.reshape(-1, 1, 64, 64)
        # print(x.shape)
        # print(c_concat[0].shape)
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out


# class Trainer(object):
#     def __init__(self, model, trainloader, args, valloader=None):
#         super(Trainer, self).__init__()
#         self.args = args
#         self.model = model
#         self.dataloader = trainloader
#         self.test_loader =  valloader




# def launch():
#     import argparse
#     parser = argparse.ArgumentParser()
#     args = parser.parse_args()
#     args.run_name = "DDPM_conditional"
#     args.epochs = 300
#     args.batch_size = 14
#     # args.image_size = 64
#     # args.num_classes = 10
#     # args.dataset_path = r"C:\Users\dome\datasets\cifar10\cifar10-64\train"
#     args.device = "cuda"
#     args.lr = 3e-4
#     train(args)
#
#
# if __name__ == '__main__':
#     launch()
    # device = "cuda"
    # model = UNet_conditional(num_classes=10).to(device)
    # ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # n = 8
    # y = torch.Tensor([6] * n).long().to(device)
    # x = diffusion.sample(model, n, y, cfg_scale=0)
    # plot_images(x)

