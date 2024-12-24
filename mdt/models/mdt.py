"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from collections import OrderedDict
from copy import deepcopy
from functools import partial

import copy
import functools
import os

import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

# from mdt.modules.masked_diffusion import dist_util, logger
# from mdt.modules.masked_diffusion.fp16_util import MixedPrecisionTrainer
from mdt.modules.masked_diffusion.nn import update_ema
from mdt.modules.masked_diffusion.resample import LossAwareSampler, UniformSampler, create_named_schedule_sampler
# from adan import Adan
from torch.distributed.optim import ZeroRedundancyOptimizer
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


from tqdm import tqdm
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from contextlib import contextmanager
from stage2.modules.ema import LitEma
from utils.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from stage1.modules.distributions import normal_kl, DiagonalGaussianDistribution






__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


class MDT(pl.LightningModule):
    # classic DITM with Gaussian diffusion, in latent space
    def __init__(self,
                 diff_config,
                 mvit_config,
                 first_stage_config,
                 cond_stage_config,
                 learning_rate,
                 schedule_sampler="uniform",
                 latent_size=8,
                 embdim=16,
                 use_ema=True,
                 first_stage_key='weights',
                 input_size=32,
                 log_every_t=10,
                 channels=1,
                 scheduler_config=None,
                 num_timesteps_cond=None,
                 cond_stage_key="dataset",
                 cond_stage_trainable=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 ckpt_path=None,
                 ignore_keys=[],
                 device='cuda',
                 monitor=None,
                 *args, **kwargs
                 ):
        super().__init__()

        # self.device = device
        self.monitor=monitor
        self.cond_stage_model = None
        # self.scale_factor=scale_factor
        self.cond_stage_trainable=cond_stage_trainable
        self.cond_stage_forward=cond_stage_forward
        self.conditioning_key=conditioning_key


        self.learning_rate = learning_rate
        self.latent_size = latent_size
        self.embdim = embdim
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        self.cond_stage_key=cond_stage_key
        self.use_ema= use_ema


        # self.batch_size=batch_size
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.input_size = input_size  # try conv?
        self.channels = channels
        self.input_shape = (-1, self.channels, self.input_size, self.input_size)
        ###################models instanciation#####################
        self.model = instantiate_from_config(mvit_config)
        # self.diffusion = create_diffusion(diff_config)
        self.diffusion = instantiate_from_config(diff_config)

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        schedule_sampler = create_named_schedule_sampler(schedule_sampler, self.diffusion)
        self.schedule_sampler = schedule_sampler or UniformSampler(self.diffusion)

        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))


        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            self.restarted_from_ckpt = True

        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)


    # # @torch.no_grad
    # def scope_ema(self, model):
    #     self.ema = deepcopy(model).to(self.device)  # Create an EMA of the model for use after training
    #     requires_grad(self.ema, False)
    #     update_ema(self.ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    #     model.train()  # important! This enables embedding dropout for classifier-free guidance
    #     self.ema.eval()  # EMA model should always be in eval mode

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

    # def on_save_checkpoint(self, checkpoint):
    #     with ema.average_parameters():
    #         checkpoint['state_dict'] = self.state_dict()
    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
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
                # for param in self.cond_stage_model.parameters():
                #     param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    @torch.no_grad()
    def get_input(self, batch, k):
       x= batch[k]
       if isinstance(x, torch.Tensor):
           x = x.to(memory_format=torch.contiguous_format).float()
       return x

    def shared_step(self, batch, **kwargs):

        # x = batch[self.first_stage_key]
        x = self.get_input(batch, self.first_stage_key)
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()
        z = z.reshape(self.input_shape)

        if self.cond_stage_key is not None:
            xc = batch[self.cond_stage_key]
            if isinstance(xc, torch.Tensor):
                xc = xc.to(self.device)
            c = self.get_learned_conditioning(xc)
        else:
            c = []
        loss = self(z, c)
        return loss



    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        z = z.reshape(self.input_shape)
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        c = self.cond_stage_model(c)
        return c

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z
        z = z.reshape(-1, self.embdim, self.latent_size, self.latent_size)
        return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    def get_losses(self, fn, model, z, t, model_kwargs):
        # model_kwargs = dict(y=y)
        loss_dict = {}

        compute_losses = functools.partial(fn,
                                           model,
                                           z,
                                           t,
                                           model_kwargs=model_kwargs,
                                           )
        mask_kwargs = model_kwargs.copy()
        mask_kwargs['enable_mask'] = True
        compute_losses_mask = functools.partial(self.diffusion.training_losses,
                                                self.model,
                                                z,
                                                t,
                                                model_kwargs=mask_kwargs, )
        return compute_losses, compute_losses_mask



    def forward(self, z, y, *args, **kwargs):
        # t = torch.randint(0, self.diffusion.num_timesteps, (z.shape[0],), device=self.device)
        t, weights = self.schedule_sampler.sample(z.shape[0], device=self.device)
        model_kwargs = dict(y=y)

        compute_losses = functools.partial(self.diffusion.training_losses,
                                           self.model,
                                           z,
                                           t,
                                           model_kwargs=model_kwargs,
                                           )
        mask_kwargs = model_kwargs.copy()
        mask_kwargs['enable_mask'] = True
        compute_losses_mask = functools.partial(self.diffusion.training_losses,
                                                self.model,
                                                z,
                                                t,
                                                model_kwargs=mask_kwargs, )
        loss_dict = {}


        losses = compute_losses()
        losses_mask = compute_losses_mask()

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach() + losses_mask["loss"].detach()
            )

        loss = (losses["loss"] * weights).mean() + (losses_mask["loss"] * weights).mean()
        ploss= {k: v * weights for k, v in losses.items()}
        mloss = {'m_' + k: v * weights for k, v in losses_mask.items()}
        loss_dict.update(ploss)
        loss_dict.update(mloss)
        # loss_dict["losses"]=loss
        return loss, loss_dict

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def training_step(self, batch, batch_idx):
        prefix='train'
        loss, loss_dict = self.shared_step(batch)
        # print(list(loss_dict))
        # loss_dict_no_ema = {key + 'no__ema': loss_dict_no_ema[key].mean() for key in loss_dict_no_ema}
        loss_dict = {key: loss_dict[key].mean() for key in loss_dict}
        loss_dict.update({f'{prefix}/loss': loss})
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True, sync_dist=True)
        # self.log('train_acc', acc, on_epoch=True, logger=True, batch_size=batch_size)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True)


        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True)

        return loss




    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, loss_dict_no_ema = self.shared_step(batch)
        prefix='val'
        # print(list(loss_dict_no_ema))
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
        #     loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        loss_dict_ema = {key+ '__ema': loss_dict_ema[key].mean() for key in loss_dict_ema}
        # loss_dict_no_ema.update({f'{prefix}/loss': loss})
        # self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)

    # def on_train_batch_end(self, *args, **kwargs):
    #     if self.use_ema:
    #         update_ema(self.ema, self.model)
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            # x = batch[self.first_stage_key]
            x = self.get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")




    def configure_optimizers(self):

        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        # if self.learn_logvar:
        #     print('Diffusion model optimizing logvar')
        #     params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:

            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler

        return opt

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def condsample(self, y, N=1, x_start=None):
        print(f'Conditionally sampling {len(y)} samples')
        if isinstance(y, list):
            if N > 1 and len(y) == 1:
                y = y * N
            y = torch.tensor(y, dtype=torch.long)
        if y.shape[0] == 1 and N > 1:
            y = y.tile(N, )
        y = y.reshape(-1,).to(self.device)
        assert isinstance(y,
                          torch.Tensor), f'Class conditioning argument should be a tensor or a list but got {type(y)} instead.'
        N = y.shape[0]
        # print(f'batch size----{N}---{y.shape}--==============================================')
        model_kwargs = dict(y=y)
        shape = (N, self.channels, self.input_size, self.input_size)
        samples = self.diffusion.p_sample_loop(
            self.model.forward, shape, x_start, clip_denoised=False, model_kwargs=model_kwargs,
            progress=True, device=self.device
        )
        samples = self.decode_first_stage(samples)
        return samples


