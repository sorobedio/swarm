import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SuperConv1D(nn.Conv1d):
    def __init__(self, super_in_dim, super_out_dim, kernel_size, stride=1, padding=0,
                 bias=False, uniform_=None, non_linear='linear', scale=False):
        super().__init__(super_in_dim, super_out_dim, kernel_size,
                         stride=stride, padding=padding, bias=bias)

        # Super dimensions represent the largest possible network
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        # Sample dimensions for current architecture
        self.sample_in_dim = None
        self.sample_out_dim = None

        self.samples = {}
        self.scale = scale
        self._reset_parameters(bias, uniform_, non_linear)
        self.profiling = False

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _reset_parameters(self, bias, uniform_, non_linear):
        nn.init.xavier_uniform_(self.weight) if uniform_ is None else uniform_(
            self.weight, non_linear=non_linear)
        if bias:
            nn.init.constant_(self.bias, 0.)

    def set_sample_config(self, sample_in_dim, sample_out_dim):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim
        self._sample_parameters()

    def _sample_parameters(self):
        self.samples['weight'] = sample_weight(self.weight, self.sample_in_dim,
                                               self.sample_out_dim)
        self.samples['bias'] = self.bias
        self.sample_scale = self.super_out_dim / self.sample_out_dim

        if self.bias is not None:
            self.samples['bias'] = sample_bias(self.bias, self.sample_out_dim)
        return self.samples

    def forward(self, x):
        self.sample_parameters()
        return F.conv1d(x,
                        self.samples['weight'],
                        self.samples['bias'],
                        self.stride,
                        self.padding) * (self.sample_scale if self.scale else 1)

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        weight_numel = self.samples['weight'].numel()

        if self.samples['bias'] is not None:
            bias_numel = self.samples['bias'].numel()
        else:
            bias_numel = 0

        return weight_numel + bias_numel

    def get_complexity(self, sequence_length):
        # For Conv1D: output_length * kernel_size * in_channels * out_channels
        output_length = (sequence_length + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        total_flops = output_length * self.kernel_size[0] * self.sample_in_dim * self.sample_out_dim
        return total_flops


def sample_weight(weight, sample_in_dim, sample_out_dim):
    # For Conv1D weight shape is (out_channels, in_channels, kernel_size)
    sample_weight = weight[:sample_out_dim, :sample_in_dim, :]
    return sample_weight


def sample_bias(bias, sample_out_dim):
    sample_bias = bias[:sample_out_dim]
    return sample_bias



def sample_conv1d_weight(weight, sample_out_channels, sample_in_channels, sample_kernel_size):
    """
    weight shape: [out_channels, in_channels, kernel_size]
    """
    weight_sampled = weight[:sample_out_channels, :sample_in_channels, :sample_kernel_size]
    return weight_sampled

def sample_conv1d_bias(bias, sample_out_channels):
    """
    bias shape: [out_channels]
    """
    bias_sampled = bias[:sample_out_channels]
    return bias_sampled

class Conv1dSuper(nn.Conv1d):
    """
    A "super" 1D convolution layer modeled after the LinearSuper approach.
    It allows for sampling sub-kernels from a super kernel (largest possible kernel).
    """

    def __init__(
        self,
        super_in_channels,
        super_out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        uniform_=None,
        non_linear='conv',
        scale=False,
    ):
        """
        Args:
            super_in_channels (int): Max (super) number of input channels
            super_out_channels (int): Max (super) number of output channels
            kernel_size (int): Max (super) kernel size
            stride (int): Conv1d stride
            padding (int): Conv1d padding
            dilation (int): Conv1d dilation
            groups (int): Conv1d groups
            bias (bool): If True, adds a learnable bias
            uniform_ (callable): Custom weight initializer (e.g., `nn.init.xavier_uniform_`)
            non_linear (str): For choosing initializer type, if you have different types in your custom init
            scale (bool): If True, multiply output by (super_out_channels / sampled_out_channels)
        """
        super().__init__(
            in_channels=super_in_channels,
            out_channels=super_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        # Record "super" dimensions
        self.super_in_channels = super_in_channels
        self.super_out_channels = super_out_channels
        self.super_kernel_size = kernel_size

        # Will be set when we choose a sub-configuration
        self.sample_in_channels = None
        self.sample_out_channels = None
        self.sample_kernel_size = None

        # For storing references to the sampled weights/bias
        self.samples = {}

        # Optional scaling of output
        self.scale = scale
        self.sample_scale = 1.0

        # Profiling mode
        self.profiling = False

        # Initialize weights
        self._reset_parameters(bias, uniform_, non_linear)

    def _reset_parameters(self, bias, uniform_, non_linear):
        """
        Reset the parameters of the super-kernel.
        By default, uses PyTorch's reset logic (kaiming_uniform_).
        If a custom init is provided, apply it.
        """
        if uniform_ is not None:
            uniform_(self.weight, non_linear=non_linear)
        else:
            # Example default init
            # You can change to nn.init.xavier_uniform_ or something else
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias and self.bias is not None:
            nn.init.constant_(self.bias, 0.)

    def profile(self, mode=True):
        """Toggle profiling mode."""
        self.profiling = mode

    def sample_parameters(self, resample=False):
        """Return the dictionary of sampled parameters."""
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def set_sample_config(self, sample_in_channels, sample_out_channels, sample_kernel_size=None):
        """
        Set how many channels and kernel size to pick from the super kernel.

        Args:
            sample_in_channels (int): number of input channels to sample
            sample_out_channels (int): number of output channels to sample
            sample_kernel_size (int): kernel size to sample
        """
        # If sample_kernel_size is not specified, assume the entire super kernel size
        if sample_kernel_size is None:
            sample_kernel_size = self.super_kernel_size

        self.sample_in_channels = sample_in_channels
        self.sample_out_channels = sample_out_channels
        self.sample_kernel_size = sample_kernel_size

        self._sample_parameters()

    def _sample_parameters(self):
        # Sample from the super weight
        self.samples['weight'] = sample_conv1d_weight(
            self.weight,
            sample_out_channels=self.sample_out_channels,
            sample_in_channels=self.sample_in_channels,
            sample_kernel_size=self.sample_kernel_size
        )

        # Sample from the super bias if present
        if self.bias is not None:
            self.samples['bias'] = sample_conv1d_bias(self.bias, self.sample_out_channels)
        else:
            self.samples['bias'] = None

        # Compute scale if requested
        if self.scale:
            self.sample_scale = self.super_out_channels / float(self.sample_out_channels)
        else:
            self.sample_scale = 1.0

        return self.samples

    def forward(self, x):
        # Make sure we have the correct sampled parameters
        self.sample_parameters()
        return F.conv1d(
            x,
            self.samples['weight'],
            bias=self.samples['bias'],
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        ) * self.sample_scale

    def calc_sampled_param_num(self):
        """
        Return the number of parameters (weight + bias) in the current sampled configuration.
        """
        assert 'weight' in self.samples, "Must sample parameters first."
        num_weight = self.samples['weight'].numel()
        num_bias = 0
        if self.samples['bias'] is not None:
            num_bias = self.samples['bias'].numel()
        return num_weight + num_bias

    def get_complexity(self, sequence_length):
        """
        Estimate complexity (FLOPs) for the sampled conv1d layer.
        This is a simplistic approach (similar to the LinearSuper code).

        For a conv1d, the FLOP count is commonly estimated as:
            out_channels * in_channels * kernel_size * output_length
            multiplied by the batch size.
        But here we'll mimic the style from LinearSuper:
            total_flops = sequence_length * np.prod(self.samples['weight'].size())

        Adjust this logic as needed.
        """
        # shape of self.samples['weight']: [sample_out_channels, sample_in_channels, sample_kernel_size]
        wsize = list(self.samples['weight'].size())  # [outC, inC, k]
        total_flops = sequence_length * np.prod(wsize)
        return total_flops
