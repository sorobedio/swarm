import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            # nn.LayerNorm(dim),
            # nn.GELU(),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            # nn.LayerNorm(dim),
            nn.Dropout(dropout)
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class LEncoder(nn.Module):
    def __init__(self, in_dim, z_features, in_channels=16,my_channels=1, double_z=True,
                 fdim=1024, mult=[1, 1, 1, 1], num_residual_blocks=2):
        super(LEncoder, self).__init__()

        self.in_dim = in_dim
        self.in_channels = in_channels
        self.fdim = fdim
        self.my_channels = my_channels

        if double_z:
            z_features = z_features * 2
        self.z_features = z_features

        # Input projection
        self.sterm = nn.Linear(in_dim, fdim)

        # Calculate dimensions for each downsampling level
        dims = [fdim]
        current_dim = fdim
        for m in mult:
            current_dim = current_dim // 2
            dims.append(current_dim * m)

        # Downsampling layers and residual blocks
        self.layers = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()

        for i in range(len(dims) - 1):
            layer = nn.Sequential(
                nn.Linear(dims[i], dims[i + 1]),
                # nn.LayerNorm(dims[i + 1]),
                nn.LeakyReLU(),
                nn.Dropout(0.1)
            )
            self.layers.append(layer)

            res_blocks = nn.ModuleList([
                ResidualBlock(dims[i + 1])
                for _ in range(num_residual_blocks)
            ])
            self.residual_blocks.append(res_blocks)

        self.final = nn.Linear(dims[-1], z_features)

    def forward(self, x):
        # print(x.shape)
        # print('=========================')
        batch_size = x.shape[0]
        # Flatten input
        x = x.view(-1, self.my_channels, self.in_dim)
        x = self.sterm(x)
        x = x.view(batch_size, self.in_channels, self.fdim)

        # Pass through layers
        for layer, res_blocks in zip(self.layers, self.residual_blocks):
            x = layer(x)
            for res_block in res_blocks:
                x = res_block(x)

        z = self.final(x)
        # print(z.shape)
        return z


class LDecoder(nn.Module):
    def __init__(self, in_dim, z_features, in_channels=16, my_channels=1, double_z=True,
                 fdim=1024, mult=[1, 1, 1, 1], num_residual_blocks=2):
        super(LDecoder, self).__init__()

        self.in_dim = in_dim
        self.in_channels = in_channels
        self.fdim = fdim
        self.my_channels=my_channels

        if double_z:
            self.z_features = z_features * 2
        else:
            self.z_features = z_features

        # Calculate upsampling dimensions
        dims = [fdim]
        current_dim = fdim
        for m in reversed(mult):
            current_dim = current_dim // 2
            dims.append(current_dim * m)
        dims = list(reversed(dims))

        # Initial layer
        self.initial = nn.Sequential(
            nn.Linear(self.z_features, dims[0]),
            nn.LeakyReLU(),
            # nn.LayerNorm(dims[0]),
            # nn.GELU(),
            nn.Dropout(0.1)
        )

        # Upsampling layers and residual blocks
        self.layers = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()

        for i in range(len(dims) - 1):
            layer = nn.Sequential(
                nn.Linear(dims[i], dims[i + 1]),
                nn.LeakyReLU(),
                # nn.LayerNorm(dims[i + 1]),
                # nn.GELU(),
                # nn.Dropout(0.1)
            )
            self.layers.append(layer)

            res_blocks = nn.ModuleList([
                ResidualBlock(dims[i + 1])
                for _ in range(num_residual_blocks)
            ])
            self.residual_blocks.append(res_blocks)

        self.out_sterm = nn.Linear(fdim, in_dim)

    def forward(self, z):
        batch_size = z.shape[0]
        x = self.initial(z)

        for layer, res_blocks in zip(self.layers, self.residual_blocks):
            x = layer(x)
            for res_block in res_blocks:
                x = res_block(x)

        # Reshape and output
        x = x.view(batch_size, self.in_channels, self.fdim)
        x = self.out_sterm(x)
        # x = x.view(batch_size, self.in_channels, self.in_dim)
        return x