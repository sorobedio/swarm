import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from stage1.modules.model  import Encoder, Decoder, MyDecoder, MyEncoder
# from modules.vqvae_module import VectorQuantizer, VectorQuantizerEMA, VqDecoder, VqEncoder
from stage1.modules.distributions import DiagonalGaussianDistribution

from utils.util import instantiate_from_config

class AutoencoderKL(nn.Module):
    def __init__(self, ddconfig,
                 lossconfig,
                 # cond_stage_config,
                 embed_dim,
                 learning_rate,
                 ckpt_path=None,
                 ignore_keys=[],
                 input_key="weight",
                 cond_key="weight",):
        super(AutoencoderKL, self).__init__()
        self.learning_rate = learning_rate
        self.ckpt_path= ckpt_path
        self.ignore_keys = ignore_keys
        self.input_key = input_key
        self.cond_key = cond_key
        self.embed_dim = embed_dim

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)

        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, 2*ddconfig["z_channels"], 1)

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




class CAutoencoderKL(nn.Module):
    def __init__(self, ddconfig,
                 lossconfig,
                 # cond_stage_config,
                 embed_dim,
                 learning_rate,
                 ckpt_path=None,
                 ignore_keys=[],
                 input_key="weight",
                 cond_key="weight",):
        super(CAutoencoderKL, self).__init__()
        self.learning_rate = learning_rate
        self.ckpt_path= ckpt_path
        self.ignore_keys = ignore_keys
        self.input_key = input_key
        self.cond_key = cond_key
        self.embed_dim = embed_dim

        self.encoder = MyEncoder(**ddconfig)
        self.decoder = MyDecoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)

        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
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
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec


    def vqencode(self, x):
        # h = self.encoder(x)
        moments = self.quant_conv(x)
        posterior = DiagonalGaussianDistribution(moments)
        z = posterior.sample()
        dec = self.decode(z)
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
                                  lr=lr, betas=(0.5, 0.9),weight_decay=5e-4)

        # opt_ae = torch.optim.SGD(list(self.encoder.parameters()) +
        #                           list(self.decoder.parameters()) +
        #                           list(self.quant_conv.parameters()) +
        #                           list(self.post_quant_conv.parameters()),
        #                           lr=lr, momentum=0.9, weight_decay=0.0)

        return opt_ae

    def get_last_layer(self):
        return self.decoder.conv_out.weight




class LAutoencoderKL(nn.Module):
    def __init__(self, ddconfig,
                 lossconfig,
                 # cond_stage_config,
                 embed_dim,
                 learning_rate,
                 ckpt_path=None,
                 ignore_keys=[],
                 input_key="weight",
                 cond_key="weight",):
        super(LAutoencoderKL, self).__init__()
        self.learning_rate = learning_rate
        self.ckpt_path= ckpt_path
        self.ignore_keys = ignore_keys
        self.input_key = input_key
        self.cond_key = cond_key
        self.embed_dim = embed_dim

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        # self.loss = instantiate_from_config(lossconfig)

        # self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        # self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
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
        # moments = self.quant_conv(h)
        # posterior = DiagonalGaussianDistribution(moments)
        return h

    def decode(self, z):
        # print(z.shape)
        # exit()
        # z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec


    # def vqencode(self, x):
    #     # h = self.encoder(x)
    #     moments = self.quant_conv(x)
    #     posterior = DiagonalGaussianDistribution(moments)
    #     z = posterior.sample()
    #     dec = self.decode(z)
    #     return dec

    def forward(self, batch):
        if isinstance(batch, dict):
            inputs = self.get_input(batch, self.input_key)
        else:
            inputs = batch
        z = self.encode(inputs)

        dec = self.decode(z)
        return dec
    def get_input(self, batch, k):
        x = batch[k]
        return x
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()),
                                  lr=lr, betas=(0.5, 0.9))

        # opt_ae = torch.optim.SGD(list(self.encoder.parameters()) +
        #                           list(self.decoder.parameters()) +
        #                           list(self.quant_conv.parameters()) +
        #                           list(self.post_quant_conv.parameters()),
        #                           lr=lr, momentum=0.9, weight_decay=0.0, nesterov=True)

        return opt_ae

    def get_last_layer(self):
        return self.decoder.conv_out.weight




# class VQVAE(nn.Module):
#     def __init__(self,
#                  # first_stage,
#                  ddconfig,
#                  num_hiddens,
#                  num_embeddings,
#                  embedding_dim,
#                  commitment_cost,
#                  learning_rate,
#                  decay=0,
#                  input_key='weight',
#                  cond_key='dataset',
#                  use_fist_stae=False
#                  ):
#         super(VQVAE, self).__init__()
#         self.input_key= input_key
#         self.cond_key= cond_key
#         self.use_first_stage = use_fist_stae
#         self.learning_rate = learning_rate
#         self.decay = decay
#         self.commitment_cost = commitment_cost
#
#         # self.first_stage_config= first_stage
#         #
#         # self.instantiate_first_stage(first_stage)
#
#         self._encoder = VqEncoder(**ddconfig)
#         self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
#                                       out_channels=embedding_dim,
#                                       kernel_size=1,
#                                       stride=1)
#         if decay > 0.0:
#             self.codebook = VectorQuantizerEMA(num_embeddings, embedding_dim,
#                                               commitment_cost, decay)
#         else:
#             self.codebook = VectorQuantizer(num_embeddings, embedding_dim,
#                                            commitment_cost)
#         self._decoder = VqDecoder(**ddconfig)
#
#     # self.restarted_from_ckpt = False
#     # if ckpt_path is not None:
#     #     self.init_from_ckpt(ckpt_path, ignore_keys)
#     #     self.restarted_from_ckpt = True
#     def init_from_ckpt(self, path, ignore_keys=list()):
#         sd = torch.load(path, map_location="cpu")["state_dict"]
#         keys = list(sd.keys())
#         for k in keys:
#             for ik in ignore_keys:
#                 if k.startswith(ik):
#                     print("Deleting key {} from state_dict.".format(k))
#                     del sd[k]
#         self.load_state_dict(sd, strict=False)
#         print(f"Restored from {path}")
#
#     # def instantiate_first_stage(self, config):
#     #     model = instantiate_from_config(config)
#     #     self.first_stage_model = model.eval()
#     #     # self.first_stage_model.train = disabled_train
#     #     for param in self.first_stage_model.parameters():
#     #         param.requires_grad = False
#
#     def get_input(self, batch, k):
#         x = batch[k]
#         return x
#
#
#     def forward(self, batch):
#         if isinstance(batch, dict):
#             inputs = self.get_input(batch, self.input_key)
#         else:
#             inputs = batch
#
#         # print(inputs.shape)
#         # x = x.reshape(-1, 1, 32, 32)
#         z = self._encoder(inputs)
#         # print(z.shape)
#         z = self._pre_vq_conv(z)
#         loss, quantized, perplexity, _, _ = self.codebook(z)
#         x_recon = self._decoder(quantized)
#
#         return loss, x_recon, perplexity
#
#     def encode(self, x):
#         z = self._encoder(x)
#         z = self._pre_vq_conv(z)
#         loss, quantized, perplexity, encodings, indices = self.codebook(z)
#         indices = indices.reshape(x.shape[0], -1)
#         return loss, quantized, perplexity, encodings, indices
#
#     def decode(self, quantized):
#         x_recon = self._decoder(quantized)
#         return x_recon
#
#
#     def configure_optimizers(self):
#         lr = self.learning_rate
#         opt_ae = torch.optim.Adam(list(self._encoder.parameters()) +
#                                   list(self._decoder.parameters())+
#                                   list(self._pre_vq_conv.parameters()) +
#                                   list(self.codebook.parameters()),
#                                   lr=lr, betas=(0.5, 0.9))
#
#         return opt_ae
#
#
#
#
#
#
#
# class VQVAEModel(nn.Module):
#     def __init__(self,
#                  # first_stage,
#                  lossconfig,
#                  ddconfig,
#                  num_hiddens,
#                  num_embeddings,
#                  embedding_dim,
#                  commitment_cost,
#                  learning_rate,
#                  decay=0,
#                  input_key='weight',
#                  cond_key='dataset',
#                  use_fist_stae=False
#                  ):
#         super(VQVAEModel, self).__init__()
#         self.input_key= input_key
#         self.cond_key= cond_key
#         self.use_first_stage = use_fist_stae
#         self.learning_rate = learning_rate
#         self.decay = decay
#         self.commitment_cost = commitment_cost
#
#         # self.first_stage_config= first_stage
#         #
#         # self.instantiate_first_stage(first_stage)
#         self.loss = instantiate_from_config(lossconfig)
#
#         self._encoder = VqEncoder(**ddconfig)
#         self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
#                                       out_channels=embedding_dim,
#                                       kernel_size=1,
#                                       stride=1)
#         if decay > 0.0:
#             self.codebook = VectorQuantizerEMA(num_embeddings, embedding_dim,
#                                               commitment_cost, decay)
#         else:
#             self.codebook = VectorQuantizer(num_embeddings, embedding_dim,
#                                            commitment_cost)
#         self._decoder = VqDecoder(**ddconfig)
#         # self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
#         self.post_quant_conv = nn.Conv2d(embedding_dim, embedding_dim*2, 1)
#
#     # self.restarted_from_ckpt = False
#     # if ckpt_path is not None:
#     #     self.init_from_ckpt(ckpt_path, ignore_keys)
#     #     self.restarted_from_ckpt = True
#     def init_from_ckpt(self, path, ignore_keys=list()):
#         sd = torch.load(path, map_location="cpu")["state_dict"]
#         keys = list(sd.keys())
#         for k in keys:
#             for ik in ignore_keys:
#                 if k.startswith(ik):
#                     print("Deleting key {} from state_dict.".format(k))
#                     del sd[k]
#         self.load_state_dict(sd, strict=False)
#         print(f"Restored from {path}")
#
#     # def instantiate_first_stage(self, config):
#     #     model = instantiate_from_config(config)
#     #     self.first_stage_model = model.eval()
#     #     # self.first_stage_model.train = disabled_train
#     #     for param in self.first_stage_model.parameters():
#     #         param.requires_grad = False
#
#     def get_input(self, batch, k):
#         x = batch[k]
#         return x
#
#
#     def forward(self, batch):
#         if isinstance(batch, dict):
#             inputs = self.get_input(batch, self.input_key)
#         else:
#             inputs = batch
#
#         # x = x.reshape(-1, 1, 32, 32)
#         # z = self._encoder(inputs)
#         # z = self._pre_vq_conv(z)
#         loss, quantized, perplexity, _, _ = self.encode(inputs)
#         x_recon, posterior = self.decode(quantized)
#
#         return loss, x_recon, perplexity, posterior
#
#     def encode(self, x):
#         z = self._encoder(x)
#         z = self._pre_vq_conv(z)
#         loss, quantized, perplexity, encodings, indices = self.codebook(z)
#
#         indices = indices.reshape(x.shape[0], -1)
#         return loss, quantized, perplexity, encodings, indices
#
#     def decode(self, quantized):
#         # print(quantized.shape)
#         moments = self.post_quant_conv(quantized)
#         # print(moments.shape)
#         posterior = DiagonalGaussianDistribution(moments)
#         z = posterior.sample()
#         # print(z.shape)
#         x_recon = self._decoder(z)
#         return x_recon, posterior
#
#
#     def configure_optimizers(self):
#         lr = self.learning_rate
#         opt_ae = torch.optim.Adam(list(self._encoder.parameters()) +
#                                   list(self._decoder.parameters())+
#                                   list(self._pre_vq_conv.parameters()) +
#                                   list(self.codebook.parameters())+
#                                   list(self.post_quant_conv.parameters()),
#                                   lr=lr, betas=(0.5, 0.9))
#
#         return opt_ae
#
#
#
#     # def encode(self, x):
#     #     h = self.encoder(x)
#     #     moments = self.quant_conv(h)
#     #     posterior = DiagonalGaussianDistribution(moments)
#     #     return posterior
#
#
#
