import torch
import numpy as np

def compute_scalar_weights(log_weights):
    """
    Convert spatial probabilistic weights into scalar weights.

    Args:
        log_weights (torch.Tensor): Log-space weights with shape (batch_size, num_mixtures, 1, H, W).

    Returns:
        torch.Tensor: Normalized scalar weights with shape (batch_size, num_mixtures, 1).
    """
    # Aggregate spatial weights by summing over spatial dimensions (H, W)
    scalar_log_weights = torch.sum(log_weights, dim=[3, 4], keepdim=False)  # Shape: (batch_size, num_mixtures, 1)

    # Normalize in log-space to ensure the weights sum to 1 across mixtures
    scalar_log_weights = scalar_log_weights - torch.logsumexp(scalar_log_weights, dim=1, keepdim=True)  # Shape: (batch_size, num_mixtures, 1)

    # Convert from log-space to probability space
    scalar_weights = torch.exp(scalar_log_weights)  # Shape: (batch_size, num_mixtures, 1)

    return scalar_weights


class MixtureGaussianDistribution:
    def __init__(self, means, logvars, log_weights, deterministic=False):
        """
        Initialize the mixture of Gaussians distribution.

        Args:
            means (torch.Tensor): Means for each mixture component, shape (batch_size, num_mixtures, C, H, W).
            logvars (torch.Tensor): Log-variances for each mixture component, shape (batch_size, num_mixtures, C, H, W).
            log_weights (torch.Tensor): Logarithm of mixture weights, shape (batch_size, num_mixtures, 1).
            deterministic (bool): Whether to ignore stochasticity (used in mode computation).
        """
        self.means = means  # Shape: (batch_size, num_mixtures, C, H, W)
        self.logvars = torch.clamp(logvars, -30.0, 30.0)  # Numerical stability
        self.log_weights = log_weights  # Shape: (batch_size, num_mixtures, 1)
        self.weights = torch.softmax(log_weights, dim=1)  # Convert to probabilities
        self.deterministic = deterministic
        self.stds = torch.exp(0.5 * self.logvars)
        self.vars = torch.exp(self.logvars)

        if self.deterministic:
            self.stds = torch.zeros_like(self.means).to(device=self.means.device)
            self.vars = torch.zeros_like(self.means).to(device=self.means.device)

    # def sample(self, n=1):
    #     """
    #     Sample from the mixture of Gaussians using the exact variance.
    #
    #     Args:
    #         n (int): Number of samples to generate per input in the batch.
    #
    #     Returns:
    #         torch.Tensor: Samples from the mixture, shape (batch_size * n, latent channels, H, W).
    #     """
    #     batch_size, num_mixtures, latent_channels, H, W = self.means.shape
    #
    #     # Ensure mixture dimensions are consistent
    #     assert self.weights.shape == (batch_size, num_mixtures,
    #                                   1), f"Expected weights shape: {(batch_size, num_mixtures, 1)}, got {self.weights.shape}"
    #     assert self.stds.shape == self.means.shape, f"Means and stds shapes must match: {self.means.shape} vs {self.stds.shape}"
    #
    #     # Expand weights for n samples per batch
    #     expanded_weights = self.weights.unsqueeze(1).expand(batch_size, n, num_mixtures,
    #                                                         1)  # Shape: (batch_size, n, num_mixtures, 1)
    #     expanded_weights = expanded_weights.reshape(-1, num_mixtures, 1)  # Shape: (batch_size * n, num_mixtures, 1)
    #
    #     # Expand means and stds for n samples per batch
    #     expanded_means = self.means.unsqueeze(1).expand(batch_size, n, num_mixtures, latent_channels, H,
    #                                                     W)  # Shape: (batch_size, n, num_mixtures, latent_channels, H, W)
    #     expanded_means = expanded_means.reshape(-1, num_mixtures, latent_channels, H,
    #                                             W)  # Shape: (batch_size * n, num_mixtures, latent_channels, H, W)
    #
    #     expanded_stds = self.stds.unsqueeze(1).expand(batch_size, n, num_mixtures, latent_channels, H,
    #                                                   W)  # Shape: (batch_size, n, num_mixtures, latent_channels, H, W)
    #     expanded_stds = expanded_stds.reshape(-1, num_mixtures, latent_channels, H,
    #                                           W)  # Shape: (batch_size * n, num_mixtures, latent_channels, H, W)
    #
    #     # Compute the weighted mean
    #     weighted_mean = torch.sum(expanded_weights * expanded_means,
    #                               dim=1)  # Shape: (batch_size * n, latent_channels, H, W)
    #
    #     # Compute within-component variance
    #     within_component_variance = torch.sum(expanded_weights * (expanded_stds ** 2),
    #                                           dim=1)  # Shape: (batch_size * n, latent_channels, H, W)
    #
    #     # Compute between-component variance
    #     between_component_variance = torch.sum(
    #         expanded_weights * (expanded_means - weighted_mean.unsqueeze(1)) ** 2, dim=1
    #     )  # Shape: (batch_size * n, latent_channels, H, W)
    #
    #     # Total variance
    #     total_variance = within_component_variance + between_component_variance  # Shape: (batch_size * n, latent_channels, H, W)
    #
    #     # Sample from the resulting Gaussian
    #     z = weighted_mean + torch.sqrt(total_variance) * torch.randn_like(
    #         weighted_mean)  # Shape: (batch_size * n, latent_channels, H, W)
    #     return z

    def sample(self, n=1):
        """
        Sample from the mixture of Gaussians using the exact variance.

        Args:
            n (int): Number of samples to generate per input in the batch.

        Returns:
            torch.Tensor: Samples from the mixture, shape (batch_size * n, latent channels, H, W).
        """
        batch_size, num_mixtures, _, H, W = self.weights.shape
        _, _, latent_channels, _, _ = self.means.shape

        # Ensure mixture dimensions are consistent
        assert self.means.shape == (batch_size, num_mixtures, latent_channels, H, W), \
            f"Expected means shape: {(batch_size, num_mixtures, latent_channels, H, W)}, got {self.means.shape}"
        assert self.stds.shape == self.means.shape, \
            f"Means and stds shapes must match: {self.means.shape} vs {self.stds.shape}"

        # Expand weights, means, and stds for n samples
        expanded_weights = self.weights.unsqueeze(1).expand(batch_size, n, num_mixtures, 1, H,
                                                            W)  # Shape: (batch_size, n, num_mixtures, 1, H, W)
        expanded_weights = expanded_weights.reshape(-1, num_mixtures, 1, H,
                                                    W)  # Shape: (batch_size * n, num_mixtures, 1, H, W)

        expanded_means = self.means.unsqueeze(1).expand(batch_size, n, num_mixtures, latent_channels, H,
                                                        W)  # Shape: (batch_size, n, num_mixtures, latent_channels, H, W)
        expanded_means = expanded_means.reshape(-1, num_mixtures, latent_channels, H,
                                                W)  # Shape: (batch_size * n, num_mixtures, latent_channels, H, W)

        expanded_stds = self.stds.unsqueeze(1).expand(batch_size, n, num_mixtures, latent_channels, H,
                                                      W)  # Shape: (batch_size, n, num_mixtures, latent_channels, H, W)
        expanded_stds = expanded_stds.reshape(-1, num_mixtures, latent_channels, H,
                                              W)  # Shape: (batch_size * n, num_mixtures, latent_channels, H, W)

        # Compute the weighted mean
        weighted_mean = torch.sum(expanded_weights * expanded_means,
                                  dim=1)  # Shape: (batch_size * n, latent_channels, H, W)

        # Compute within-component variance
        within_component_variance = torch.sum(expanded_weights * (expanded_stds ** 2),
                                              dim=1)  # Shape: (batch_size * n, latent_channels, H, W)

        # Compute between-component variance
        between_component_variance = torch.sum(
            expanded_weights * (expanded_means - weighted_mean.unsqueeze(1)) ** 2, dim=1
        )  # Shape: (batch_size * n, latent_channels, H, W)

        # Total variance
        total_variance = within_component_variance + between_component_variance  # Shape: (batch_size * n, latent_channels, H, W)

        # Sample from the resulting Gaussian
        z = weighted_mean + torch.sqrt(total_variance) * torch.randn_like(
            weighted_mean)  # Shape: (batch_size * n, latent_channels, H, W)
        return z

    def mode(self):
        """
        Compute the mode of the mixture distribution.

        Returns:
            torch.Tensor: Mode of the mixture, shape (batch_size, C, H, W).
        """
        # Select the means of the component with the highest weight
        max_weight_indices = torch.argmax(self.weights, dim=1, keepdim=True)  # Shape: (batch_size, 1)
        mode = torch.gather(self.means, 1, max_weight_indices.unsqueeze(-1).expand(-1, -1, *self.means.shape[2:]))
        return mode.squeeze(1)

    def kl(self, other=None):
        """
        Compute the KL divergence between this mixture and another distribution.

        Args:
            other (MixtureGaussianDistribution): Another distribution to compute KL divergence with.
                                                 If None, compute KL divergence with a standard Gaussian.

        Returns:
            torch.Tensor: KL divergence, shape (batch_size,).
        """
        if self.deterministic:
            return torch.Tensor([0.]).to(self.means.device)

        # Compute exact weighted mean
        weighted_mean = torch.sum(self.weights * self.means, dim=1)  # Shape: (batch_size, C, H, W)

        # Compute exact total variance
        within_component_variance = torch.sum(self.weights * self.stds ** 2, dim=1)  # Shape: (batch_size, C, H, W)
        between_component_variance = torch.sum(
            self.weights * (self.means - weighted_mean.unsqueeze(1)) ** 2, dim=1
        )  # Shape: (batch_size, C, H, W)
        total_variance = within_component_variance + between_component_variance  # Shape: (batch_size, C, H, W)

        # Compute KL divergence with a standard Gaussian (mean=0, variance=1)
        kl_div = 0.5 * torch.sum(
            total_variance + weighted_mean ** 2 - 1 - torch.log(total_variance + 1e-8), dim=[1, 2, 3]
        )
        return kl_div

    def nll(self, sample, dims=[1, 2, 3]):
        """
        Compute the negative log-likelihood of a sample.

        Args:
            sample (torch.Tensor): Sample to compute NLL for, shape (batch_size, C, H, W).
            dims (list): Dimensions to sum over.

        Returns:
            torch.Tensor: Negative log-likelihood, shape (batch_size,).
        """
        if self.deterministic:
            return torch.Tensor([0.]).to(sample.device)

        # Compute log probabilities for each mixture component
        logtwopi = np.log(2.0 * np.pi)
        log_probs = -0.5 * (
            logtwopi + self.logvars + (sample.unsqueeze(1) - self.means) ** 2 / self.vars
        )
        log_probs = torch.sum(log_probs, dim=dims)

        # Compute total log probability (log-sum-exp trick for mixture)
        log_mixture_probs = log_probs + self.log_weights.squeeze(-1)
        total_log_probs = torch.logsumexp(log_mixture_probs, dim=1)

        return -total_log_probs