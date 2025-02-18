import torch
import torch.nn as nn
import torch.nn.functional as F

def chunk_wise_recon_loss(target, output, step_size=1024):
    """
    Calculate MSE loss with chunk-wise normalization.

    Parameters:
    - target: The target tensor.
    - output: The output tensor.
    - step_size: The size of each chunk.

    Returns:
    - loss: The calculated loss.
    """
    criterion = nn.MSELoss()
    loss = torch.tensor(0.0, device=output.device).float()

    # Flatten the tensors if they have more than 2 dimensions
    if len(output.shape) > 2:
        output = torch.flatten(output, start_dim=1)
        target = torch.flatten(target, start_dim=1)

    n = output.shape[0]
    m = n/step_size
    for i in range(0, n, step_size):
        start_idx = i
        end_idx = min(start_idx + step_size, n)
        tar_tmp = target[:, start_idx:end_idx]
        out_tmp = output[:, start_idx:end_idx]
        loss_tmp = criterion(tar_tmp, out_tmp)
        loss_tmp /= m
        loss += loss_tmp

    return loss

class NopaddingLoss(nn.Module):
    def __init__(self, logvar_init=0.0, kl_weight=1.0, pad_value=0):
        super().__init__()
        self.kl_weight = kl_weight
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.pad_value = pad_value

    def forward(self, inputs, reconstructions, posteriors, split="train", weights=None):
        inputs = inputs.reshape(reconstructions.shape)
        mask = (inputs != self.pad_value).float()

        # Basic reconstruction loss
        rec_loss =(inputs.contiguous() - reconstructions.contiguous())**2
        masked_rec_loss = rec_loss * mask

        # Calculate number of valid elements per batch item
        num_valid = mask.sum(dim=list(range(1, len(mask.shape)))).clamp(min=1)

        # NLL loss with masking
        # nll_loss = masked_rec_loss / torch.exp(self.logvar) + self.logvar * mask
        nll_loss = masked_rec_loss

        # Proper reduction over non-padded elements
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
            weighted_nll_loss = (
                        weighted_nll_loss.sum(dim=list(range(1, len(weighted_nll_loss.shape)))) / num_valid).mean()
        else:
            weighted_nll_loss = (nll_loss.sum(dim=list(range(1, len(nll_loss.shape)))) / num_valid).mean()

        nll_loss = (nll_loss.sum(dim=list(range(1, len(nll_loss.shape)))) / num_valid).mean()

        # KL loss
        kl_loss = posteriors.kl()
        kl_loss = kl_loss.mean()  # Changed from sum/shape[0] to mean()

        # Total loss
        loss = weighted_nll_loss + self.kl_weight * kl_loss

        # Logging with proper masking
        with torch.no_grad():
            mean_rec_loss = (masked_rec_loss.sum(dim=list(range(1, len(masked_rec_loss.shape)))) / num_valid).mean()

        log = {
            f"{split}/total_loss": loss.detach(),
            f"{split}/logvar": self.logvar.detach(),
            f"{split}/kl_loss": kl_loss.detach(),
            f"{split}/nll_loss": nll_loss.detach(),
            f"{split}/rec_loss": mean_rec_loss,
        }
        return loss, log


 # Define Log-Cosh Loss
def log_cosh_loss(y_pred, y_true):
    return torch.mean(torch.log(torch.cosh(y_pred - y_true)))

# Example usage in a training loop
# reconstruction_loss = log_cosh_loss(predicted_output, target_output)


class Myloss(nn.Module):
    def __init__(self, logvar_init=0.0, kl_weight=1.0):

        super().__init__()
        self.kl_weight = kl_weight
        # self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.huber = torch.nn.SmoothL1Loss(beta=0.05, reduction="mean")  # Huber loss

    def forward(self, inputs, reconstructions, posteriors, split="train",weights=1.0):
        inputs = inputs.reshape(reconstructions.shape)
        # mask = (inputs != self.pad_value).float()
        # rec_loss = torch.abs(inputs.contiguous() -/ reconstructions.contiguous())
        # rec_loss = (inputs.contiguous() - reconstructions.contiguous())**2
        # rec_loss=  F.smooth_l1_loss(reconstructions, inputs, reduction='mean')*1000.0
        rec_loss =   F.mse_loss(reconstructions, inputs, reduction="sum")*weights+chunk_wise_recon_loss(inputs, reconstructions, step_size=1024)
        # loss = F.mse_loss(reconstructions, inputs, reduction="mean")
        # self.logvar.data.clamp_(min=-30, max=30)

        # nll_loss = rec_loss / (torch.exp(self.logvar)*2) + self.logvar*0.5
        # rec_loss += self.huber(reconstructions, inputs)+weights+chunk_wise_recon_loss(inputs, reconstructions, step_size=1024)
        nll_loss = rec_loss
        weighted_nll_loss = nll_loss
        # if weights is not None:
        #     weighted_nll_loss = weights*nll_loss
        # weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        # nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        # kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss = weighted_nll_loss + self.kl_weight * kl_loss.mean()
        # loss = 100*F.mse_loss(inputs.contiguous(), reconstructions.contiguous()) +self.kl_weight * kl_loss.mean()

        log = {"{}/total_loss".format(split): loss.clone().detach(),
               # "{}/logvar".format(split): self.logvar.detach(),
               "{}/kl_loss".format(split): kl_loss.detach().mean()*self.kl_weight,
               "{}/nll_loss".format(split): nll_loss.detach(),
               "{}/rec_loss".format(split): rec_loss.detach(),
               }

        return loss, log




class cshloss(nn.Module):
    def __init__(self, logvar_init=0.0, kl_weight=1.0):

        super().__init__()
        self.kl_weight = kl_weight
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def forward(self, inputs, reconstructions, posteriors, split="train",weights=1000.0):
        reconstructions = reconstructions.reshape(inputs.shape)
        # rec_loss = (inputs.contiguous() - reconstructions.contiguous())**2
        rec_loss =  log_cosh_loss(reconstructions, inputs)
        self.logvar.data.clamp_(min=-30, max=30)

        # nll_loss = rec_loss / (torch.exp(self.logvar)*2) + self.logvar*0.5
        nll_loss = rec_loss
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        # weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        # nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        # kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss = weighted_nll_loss + self.kl_weight * kl_loss.mean()
        # loss = 100*F.mse_loss(inputs.contiguous(), reconstructions.contiguous()) +self.kl_weight * kl_loss.mean()

        log = {"{}/total_loss".format(split): loss.clone().detach(),
               "{}/logvar".format(split): self.logvar.detach(),
               "{}/kl_loss".format(split): kl_loss.detach().mean()*self.kl_weight,
               "{}/nll_loss".format(split): nll_loss.detach()*1000.0,
               "{}/rec_loss".format(split): rec_loss.detach()*1000.0,
               }

        return loss, log



class My_loss(nn.Module):
    def __init__(self, logvar_init=0.0, kl_weight=1.0):

        super().__init__()
        self.kl_weight = kl_weight
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def forward(self, inputs, reconstructions, posteriors, split="train",weights=None,step=8192):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        # rec_loss = (inputs.contiguous() - reconstructions.contiguous())**2

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        # nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss = weighted_nll_loss + self.kl_weight * kl_loss

        # log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
        #        "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
        #        "{}/rec_loss".format(split): rec_loss.detach().mean(),
        #        }
        return loss




class LayerWiseReconLoss(nn.Module):
    """
    MSE w/ layer-wise normalization
    """

    def __init__(self, config_path, step_size=1024):
        super(LayerWiseReconLoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.step_size=step_size
        self.loss_mean = None
        self.layer_info = torch.load(config_path)
        # layers = list(self.layer_info)
        # print(layers)

    def forward(self, output, target):
        # check validity
        assert (
            output.shape == target.shape
        ), f"Input shape mismatch. output {output.shape} vs target {target.shape}"

        loss = torch.tensor(0.0, device=output.device).float()

        layers = list(self.layer_info)
        # print(layers)
        for l in layers:
            start_idx = self.layer_info[l]['idx_start']
            end_idx = self.layer_info[l]['idx_end']
            tar_tmp = target[:, start_idx:end_idx]
            out_tmp = output[:, start_idx:end_idx]
            loss_tmp = self.criterion(out_tmp, tar_tmp)
            loss_tmp /= output.shape[0]
            loss += loss_tmp


        return loss



class ChunkWiseReconLoss(nn.Module):
    """
    MSE w/ layer-wise normalization
    """

    def __init__(self, step_size=1024):
        super(ChunkWiseReconLoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.step_size=step_size
        self.loss_mean = None
        # self.layer_info = torch.load(config_path)

    def forward(self, target, output):
        # check validity

        loss = torch.tensor(0.0, device=output.device).float()
        if len(output.shape)>2:
            output = torch.flatten(output, start_dim=1)
            target = torch.flatten(target, start_dim=1)

        # layers = list(self.layer_info)
        n= output.shape[0]
        for i in range(n, self.step_size):
            start_idx = i
            end_idx = start_idx+self.step_size
            tar_tmp = target[:, start_idx:end_idx]
            out_tmp = output[:, start_idx:end_idx]
            loss_tmp = self.criterion(tar_tmp, out_tmp)
            loss_tmp /= output.shape[0]
            loss += loss_tmp


        return loss


