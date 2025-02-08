import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SimplePerceptualLoss(nn.Module):
    def __init__(self, layers=None, use_l1=True):
        """
        Args:
            layers (list of str): The names of the VGG layers to use for the perceptual loss.
                Default is ['relu1_2', 'relu2_2', 'relu3_3'].
            use_l1 (bool): If True, use L1 loss; otherwise, use L2 loss.
        """
        super(SimplePerceptualLoss, self).__init__()
        if layers is None:
            layers = ['relu1_2', 'relu2_2', 'relu3_3']
        self.layers = layers
        self.use_l1 = use_l1

        # Load pretrained VGG16 and freeze parameters.
        vgg = models.vgg16(pretrained=True).features
        self.vgg_layers = vgg.eval()
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        # Mapping layer names to indices in VGG16's features
        self.layer_name_mapping = {
            'relu1_1': 0,
            'relu1_2': 2,
            'relu2_1': 5,
            'relu2_2': 7,
            'relu3_1': 10,
            'relu3_2': 12,
            'relu3_3': 14,
            'relu4_1': 17,
            'relu4_2': 19,
            'relu4_3': 21,
            'relu5_1': 24,
            'relu5_2': 26,
            'relu5_3': 28,
        }
        # Get the indices of the layers we want to use.
        self.selected_layer_ids = [self.layer_name_mapping[layer] for layer in self.layers]

    def forward(self, x, y):
        """
        Computes the perceptual loss between x and y.

        Args:
            x (Tensor): Generated images (batch, channels, height, width).
            y (Tensor): Target images (batch, channels, height, width).

        Returns:
            Tensor: The computed perceptual loss.
        """
        loss = 0.0
        x_in, y_in = x, y
        for i, layer in enumerate(self.vgg_layers):
            x_in = layer(x_in)
            y_in = layer(y_in)
            if i in self.selected_layer_ids:
                if self.use_l1:
                    loss += F.l1_loss(x_in, y_in)
                else:
                    loss += F.mse_loss(x_in, y_in)
        return loss


# # Example usage:
# if __name__ == "__main__":
#     # Dummy inputs (e.g., batch of 3-channel images of size 224x224)
#     x = torch.randn(4, 3, 224, 224)
#     y = torch.randn(4, 3, 224, 224)
#
#     perceptual_loss = SimplePerceptualLoss(layers=['relu1_2', 'relu2_2', 'relu3_3'])
#     loss_val = perceptual_loss(x, y)
#     print("Perceptual Loss:", loss_val.item())
