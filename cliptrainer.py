import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import clip  # pip install git+https://github.com/openai/CLIP.git


# =============================================================================
# Utility functions and modules
# =============================================================================

def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=2):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout=0.0, temb_channels=0):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.temb_proj = nn.Linear(temb_channels, out_channels) if temb_channels > 0 else None
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # If the channels do not match, create a shortcut layer.
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) \
            if in_channels != out_channels else None

    def forward(self, x, temb):
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)
        if self.temb_proj is not None and temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.shortcut is not None:
            x = self.shortcut(x)
        return x + h


def make_attn(in_channels, attn_type="vanilla"):
    # For simplicity, we simply use an identity (no attention).
    return nn.Identity()


# =============================================================================
# Adapted Decoder (mapping latent vector to an image)
# =============================================================================

class AdaptedDecoder(nn.Module):
    def __init__(self, latent_dim, *, ch, out_ch, ch_mult=(1, 2, 4, 2),
                 num_res_blocks=2, attn_resolutions=(28,), dropout=0.0,
                 resamp_with_conv=True, resolution=224, z_channels=16,
                 give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        """
        Adapted decoder that maps a latent vector to an image of shape
        (out_ch, resolution, resolution). It first maps the latent vector into
        a spatial tensor and then applies a series of ResNet and upsampling blocks.
        """
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.latent_dim = latent_dim
        self.z_channels = z_channels
        self.resolution = resolution
        self.tanh_out = tanh_out
        self.give_pre_end = give_pre_end

        # Determine the lowest resolution (curr_res) after downsampling.
        self.num_resolutions = len(ch_mult)
        curr_res = resolution // (2 ** (self.num_resolutions - 1))
        self.curr_res = curr_res

        # Map the latent vector to a spatial latent tensor.
        self.fc = nn.Linear(latent_dim, z_channels * curr_res * curr_res)

        # Determine starting channel count.
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]

        # Project spatial latent tensor to starting channels.
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # Middle processing blocks.
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=0, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=0, dropout=dropout)

        # Upsampling blocks.
        self.up = nn.ModuleList()
        curr_res_ = curr_res
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                         temb_channels=0, dropout=dropout))
                block_in = block_out
                if curr_res_ in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up_module = nn.Module()
            up_module.block = block
            up_module.attn = attn
            if i_level != 0:
                up_module.upsample = Upsample(block_in, resamp_with_conv)
                curr_res_ = curr_res_ * 2
            self.up.insert(0, up_module)  # Prepend to maintain order
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, latent):
        batch_size = latent.shape[0]
        # Map latent vector to a spatial latent tensor.
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


# =============================================================================
# CLIPViTEncoder (using CLIP’s ViT-B/32)
# =============================================================================

class CLIPViTEncoder(nn.Module):
    def __init__(self, latent_dim, input_channels, clip_model_name="ViT-B/32", device="cpu"):
        """
        Encoder based on CLIP's ViT-B/32. It modifies the initial convolution to
        accept a dataset-dependent number of channels and then maps the feature
        vector to the latent distribution parameters.
        """
        super(CLIPViTEncoder, self).__init__()
        self.latent_dim = latent_dim

        # Load the CLIP model (preprocessing is not needed here).
        self.clip_model, _ = clip.load(clip_model_name, device=device)

        # Replace the initial patch embedding to accept input_channels.
        original_conv = self.clip_model.visual.conv1
        self.clip_model.visual.conv1 = nn.Conv2d(
            in_channels=input_channels,
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

        # The output feature dimension (e.g. 512 for ViT-B/32).
        self.feature_dim = self.clip_model.visual.output_dim

        # Linear layers to compute mean and log-variance.
        self.fc_mu = nn.Linear(self.feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_dim, latent_dim)

    def forward(self, x):
        features = self.clip_model.visual(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar


# =============================================================================
# VAE model (combining encoder and decoder)
# =============================================================================

class VAE(nn.Module):
    def __init__(self, latent_dim, input_channels, out_channels, device="cpu",
                 decoder_kwargs=None, clip_model_name="ViT-B/32"):
        """
        VAE that integrates the CLIPViTEncoder and the AdaptedDecoder.
        """
        super(VAE, self).__init__()
        self.encoder = CLIPViTEncoder(latent_dim, input_channels,
                                      clip_model_name=clip_model_name, device=device)
        if decoder_kwargs is None:
            decoder_kwargs = {
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
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


# =============================================================================
# Loss function and training loop
# =============================================================================

def vae_loss(recon, x, mu, logvar):
    """
    Compute the VAE loss as a sum of reconstruction loss (MSE) and the KL divergence.
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    # KL divergence loss
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss


def train_vae(model, dataloader, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_recon = 0.0
        running_kl = 0.0
        for batch_idx, data in enumerate(dataloader):
            # If data is a tuple (e.g. images, labels), use only images.
            images = data[0].to(device) if isinstance(data, (list, tuple)) else data.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(images)
            loss, recon_loss, kl_loss = vae_loss(recon, images, mu, logvar)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_recon += recon_loss.item()
            running_kl += kl_loss.item()
        avg_loss = running_loss / len(dataloader)
        avg_recon = running_recon / len(dataloader)
        avg_kl = running_kl / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, "
              f"Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}")


# =============================================================================
# Main: Instantiate model, create dummy data, and train
# =============================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_dim = 1024
    input_channels = 4 # For example, RGB images.
    out_channels = 4
    model = VAE(latent_dim, input_channels, out_channels, device=device)
    model.to(device)

    # For demonstration, create a dummy dataset of 100 random images (224x224, 3 channels).
    dummy_data = torch.randn(100, input_channels, 224, 224)
    dummy_dataset = TensorDataset(dummy_data)
    dataloader = DataLoader(dummy_dataset, batch_size=8, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 5

    print("Starting training...")
    train_vae(model, dataloader, optimizer, device, num_epochs=num_epochs)
