import torch
import torch.nn as nn
import torch.nn.functional as F
import clip  # pip install git+https://github.com/openai/CLIP.git

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import ConvNeXt_Base_Weights, convnext_base
import clip

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# form utils.util import

from stage1.modules.attention import LinearAttention
from stage1.modules.distributions import DiagonalGaussianDistribution


class SwiGLU(nn.Module):
    def __init__(self):
        """
        SwiGLU activation function module.
        Applies Swish activation followed by a gating mechanism.
        """
        super(SwiGLU, self).__init__()

    def forward(self, x):
        """
        Forward pass for SwiGLU.

        Args:
            x1 (torch.Tensor): Input tensor for the main linear transformation.
            x2 (torch.Tensor): Input tensor for the gating mechanism.

        Returns:
            torch.Tensor: Output tensor after applying SwiGLU.
        """
        return F.silu(x) * x

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=2):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)


class Model(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, use_timestep=True, use_linear_attn=False, attn_type="vanilla"):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.use_timestep = use_timestep
        if self.use_timestep:
            # timestep embedding
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch,
                                self.temb_ch),
                torch.nn.Linear(self.temb_ch,
                                self.temb_ch),
            ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t=None, context=None):
        #assert x.shape[2] == x.shape[3] == self.resolution
        if context is not None:
            # assume aligned context, cat along channel axis
            x = torch.cat((x, context), dim=1)
        if self.use_timestep:
            # timestep embedding
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def get_last_layer(self):
        return self.conv_out.weight



class CLIPViTEncoder(nn.Module):
    def __init__(self, latent_dim, in_channels, clip_model_name="ViT-B/32", device="cpu",  **ignore_kwargs):
        """
        Encoder based on CLIP's ViT-B/32 model.

        Args:
            latent_dim (int): Dimensionality of the latent space.
            input_channels (int): Number of channels in the input images.
            clip_model_name (str): Name of the CLIP model variant to use.
            device (str): Device on which to load the model.
        """
        super(CLIPViTEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.clip_model_name = clip_model_name
        self.device = device

        # Load the CLIP model (we don't need the preprocess here).
        self.clip_model, _ = clip.load(clip_model_name, device=device)

        # For ViT, the image is split into patches via a convolution (named conv1).
        # Replace conv1 so that it accepts the dataset-specific number of channels.
        original_conv = self.clip_model.visual.conv1
        self.clip_model.visual.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=(original_conv.bias is not None)
        )
        nn.init.kaiming_normal_(self.clip_model.visual.conv1.weight, mode='fan_out', nonlinearity='relu')
        if self.clip_model.visual.conv1.bias is not None:
            nn.init.constant_(self.clip_model.visual.conv1.bias, 0)

        # Freeze all parameters in the visual encoder except the new conv1.
        for name, param in self.clip_model.visual.named_parameters():
            if "conv1" not in name:
                param.requires_grad = False

        # The output feature dimension from the visual encoder (typically 512 for ViT-B/32)
        self.feature_dim = self.clip_model.visual.output_dim

        # Linear layers to map the feature vector to latent mean and log-variance.
        self.fc_mu = nn.Linear(self.feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_dim, latent_dim)

    def forward(self, x):
        """
        Forward pass of the encoder.

        Args:
            x (Tensor): Input tensor of shape (batch, input_channels, H, W).

        Returns:
            mu (Tensor): Mean of the latent distribution, shape (batch, latent_dim).
            logvar (Tensor): Log-variance of the latent distribution, shape (batch, latent_dim).
        """
        # Obtain a feature representation (the CLS token) from the CLIP visual encoder.
        features = self.clip_model.visual(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar


# -----------------------------------------------------------------------------
# Adapted Decoder Class
# -----------------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self,  *,latent_dim, ch, out_ch, ch_mult=(1, 2, 4, 2), num_res_blocks=2,
                 attn_resolutions=(28,), dropout=0.0, resamp_with_conv=True, resolution=224,
                 z_channels=16, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        """
        Adapted decoder that maps a latent vector (of size latent_dim) into an output image
        of shape (out_ch, resolution, resolution). It is based on your provided decoder class,
        with an added initial linear mapping from the latent vector to a spatial tensor.

        Args:
            latent_dim (int): Dimensionality of the latent vector.
            ch (int): Base channel count.
            out_ch (int): Number of output channels (e.g. 3 for RGB).
            ch_mult (tuple): Multipliers for channel count at each resolution.
            num_res_blocks (int): Number of ResNet blocks per resolution.
            attn_resolutions (tuple): Resolutions (spatial sizes) at which to apply attention.
            dropout (float): Dropout rate.
            resamp_with_conv (bool): Whether to use convolution when upsampling.
            resolution (int): Desired output image resolution (assumed square).
            z_channels (int): Number of channels for the intermediate spatial latent.
            give_pre_end (bool): If True, returns the features before the final conv layer.
            tanh_out (bool): If True, apply tanh to the final output.
            use_linear_attn (bool): If True, use linear attention (if implemented).
            attn_type (str): Type of attention to use ("vanilla", "linear", or "none").
        """
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.latent_dim = latent_dim
        self.z_channels = z_channels
        self.resolution = resolution
        self.tanh_out = tanh_out
        self.ch =ch
        self.out_ch = out_ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks

        self.give_pre_end = give_pre_end

        # Compute the spatial resolution (curr_res) at the lowest level.
        self.num_resolutions = len(ch_mult)
        curr_res = resolution // (2 ** (self.num_resolutions - 1))
        self.curr_res = curr_res

        # Map the latent vector to a spatial latent tensor.
        self.fc = nn.Linear(latent_dim, z_channels * curr_res * curr_res)

        # Determine initial channel count for the decoder.
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]

        # Map the spatial latent (z_channels) to the starting block channels.
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # Middle blocks.
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=0,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=0,
                                       dropout=dropout)

        # Upsampling blocks.
        self.up = nn.ModuleList()
        curr_res_ = curr_res  # starting resolution
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=0,
                                         dropout=dropout))
                block_in = block_out
                if curr_res_ in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up_module = nn.Module()
            up_module.block = block
            up_module.attn = attn
            if i_level != 0:
                up_module.upsample = Upsample(block_in, resamp_with_conv)
                curr_res_ = curr_res_ * 2
            self.up.insert(0, up_module)  # prepend to maintain order

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, latent):
        """
        Forward pass: maps latent vector to an image.

        Args:
            latent (Tensor): Tensor of shape (batch, latent_dim).

        Returns:
            Tensor: Reconstructed image of shape (batch, out_ch, resolution, resolution).
        """
        batch_size = latent.shape[0]
        # Map latent to spatial latent tensor.
        z = self.fc(latent)
        z = z.view(batch_size, self.z_channels, self.curr_res, self.curr_res)

        h = self.conv_in(z)
        h = self.mid.block_1(h, None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, None)
        for up_module in self.up:
            for i in range(len(up_module.block)):
                h = up_module.block[i](h, None)
                if i < len(up_module.attn):
                    h = up_module.attn[i](h)
            if hasattr(up_module, 'upsample'):
                h = up_module.upsample(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h



class CLIPViTEncoder(nn.Module):
    def __init__(self, latent_dim, in_channels, clip_model_name="ViT-B/32", device="cpu", ):
        """
        Encoder based on CLIP's ViT-B/32 model.

        Args:
            latent_dim (int): Dimensionality of the latent space.
            input_channels (int): Number of channels in the input images.
            clip_model_name (str): Name of the CLIP model variant to use.
            device (str): Device on which to load the model.
        """
        super(CLIPViTEncoder, self).__init__()
        self.latent_dim = latent_dim

        # Load the CLIP model and discard the preprocessing pipeline.
        self.clip_model, _ = clip.load(clip_model_name, device=device)

        # For ViT models, the first layer is a conv for patch embedding.
        # Replace it so that it accepts input_channels instead of the default (typically 3).
        original_conv = self.clip_model.visual.conv1
        self.clip_model.visual.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=(original_conv.bias is not None)
        )
        nn.init.kaiming_normal_(self.clip_model.visual.conv1.weight, mode='fan_out', nonlinearity='relu')
        if self.clip_model.visual.conv1.bias is not None:
            nn.init.constant_(self.clip_model.visual.conv1.bias, 0)

        # Freeze all parameters in the visual encoder except the modified conv1.
        for name, param in self.clip_model.visual.named_parameters():
            if "conv1" not in name:
                param.requires_grad = False

        # The CLIP visual encoder returns a feature vector (e.g. 512-d for ViT-B/32).
        self.feature_dim = self.clip_model.visual.output_dim

        # Linear layers to produce latent distribution parameters.
        self.fc_mu = nn.Linear(self.feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_dim, latent_dim)

    def forward(self, x):
        """
        Forward pass of the encoder.

        Args:
            x (Tensor): Input image tensor of shape (batch, input_channels, H, W).

        Returns:
            Tuple[Tensor, Tensor]: (mu, logvar) each of shape (batch, latent_dim).
        """
        features = self.clip_model.visual(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar


# -----------------------------------------------------------------------------
# Combined VAE Using CLIPViTEncoder and AdaptedDecoder
# -----------------------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim, input_channels, out_channels, device="cpu",
                 decoder_kwargs=None, clip_model_name="ViT-B/32"):
        """
        VAE that uses the CLIPViTEncoder and the AdaptedDecoder.

        Args:
            latent_dim (int): Dimensionality of the latent space.
            input_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels in the output image.
            device (str): Device to use ("cpu" or "cuda").
            decoder_kwargs (dict): Optional keyword arguments to pass to the AdaptedDecoder.
            clip_model_name (str): Name of the CLIP model variant.
        """
        super(VAE, self).__init__()
        self.encoder = CLIPViTEncoder(latent_dim, input_channels, clip_model_name=clip_model_name, device=device)
        if decoder_kwargs is None:
            decoder_kwargs = {
                'clip_model_name': "ViT-B/32",
                "latent_dim": 1024,
                "input_channel": 4,
                'out_channels': 4,
                "ch": 64,
                "ch_mult": (1, 2, 4, 2),
                "num_res_blocks": 2,
                "attn_resolutions": (28,),  # adjust as needed
                "dropout": 0.0,
                "resamp_with_conv": True,
                "resolution": 224,
                "z_channels": 16,
                "give_pre_end": False,
                "tanh_out": False,
                "use_linear_attn": False,
                "attn_type": "vanilla"
            }
        self.decoder = AdaptedDecoder(latent_dim, out_ch=out_channels, **decoder_kwargs)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Complete forward pass.

        Args:
            x (Tensor): Input image tensor.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: (reconstructed image, mu, logvar)
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


# # -----------------------------------------------------------------------------
# # Example Usage
# # -----------------------------------------------------------------------------
# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     latent_dim = 128
#     input_channels = 3  # e.g. RGB images
#     out_channels = 3
#     batch_size = 4
#     dummy_input = torch.randn(batch_size, input_channels, 224, 224, device=device)
#
#     model = VAE(latent_dim, input_channels, out_channels, device=device)
#     model.to(device)
#     recon, mu, logvar = model(dummy_input)
#     print("Reconstructed image shape:", recon.shape)


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_channels, output_size=(224, 224)):
        """
        Decoder that maps a latent vector to an image matching the input dimensions.

        Args:
            latent_dim (int): Dimensionality of the latent space.
            output_channels (int): Number of channels in the output image.
            output_size (tuple): Target output image size (height, width).
        """
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.output_size = output_size

        # Project the latent vector into a spatial feature map.
        # Here we choose a feature map of size (256, 14, 14) (adjustable as needed).
        self.fc_decode = nn.Linear(latent_dim, 256 * 14 * 14)

        # Upsampling via a series of ConvTranspose2d layers:
        # 14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 28x28 -> 56x56
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 56x56 -> 112x112
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),  # 112x112 -> 224x224
            nn.Sigmoid()  # Assuming output images are normalized in [0, 1]
        )

    def forward(self, z):
        """
        Forward pass of the decoder.

        Args:
            z (Tensor): Latent vector of shape (batch, latent_dim).

        Returns:
            recon (Tensor): Reconstructed image of shape (batch, output_channels, H, W).
        """
        x = self.fc_decode(z)
        # Reshape to (batch, 256, 14, 14)
        x = x.view(-1, 256, 14, 14)
        recon = self.decoder(x)
        return recon


class VAE(nn.Module):
    def __init__(self, latent_dim, input_channels, output_channels, clip_model_name="ViT-B/32", device="cpu"):
        """
        Variational Autoencoder (VAE) that integrates the separate encoder and decoder.

        Args:
            latent_dim (int): Latent space dimensionality.
            input_channels (int): Number of channels in the input image.
            output_channels (int): Number of channels in the output image.
            clip_model_name (str): CLIP model variant to use.
            device (str): Device to load the model.
        """
        super(VAE, self).__init__()
        self.encoder = CLIPViTEncoder(latent_dim, input_channels, clip_model_name, device)
        self.decoder = Decoder(latent_dim, output_channels)

    def reparameterize(self, mu, logvar):
        """
        Applies the reparameterization trick to sample from N(mu, var).

        Args:
            mu (Tensor): Mean of the latent distribution.
            logvar (Tensor): Log-variance of the latent distribution.

        Returns:
            z (Tensor): Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Complete forward pass through the VAE.

        Args:
            x (Tensor): Input tensor of shape (batch, input_channels, H, W).

        Returns:
            recon (Tensor): Reconstructed image.
            mu (Tensor): Mean of the latent distribution.
            logvar (Tensor): Log-variance of the latent distribution.
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


# Example usage:
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_dim = 128  # User-controlled latent space dimension.
    input_channels = 3  # For example, RGB images.
    output_channels = 3  # Reconstruction should match the input channels.
    batch_size = 4
    # Create a random input tensor matching the expected input size (224x224)
    input_tensor = torch.randn(batch_size, input_channels, 224, 224, device=device)

    model = VAE(latent_dim=latent_dim, input_channels=input_channels,
                output_channels=output_channels, clip_model_name="ViT-B/32", device=device)
    model.to(device)
    recon, mu, logvar = model(input_tensor)
    print("Reconstruction shape:", recon.shape)
    print("Latent mean shape:", mu.shape)
