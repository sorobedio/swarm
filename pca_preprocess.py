import torch
import numpy as np
from sklearn.decomposition import PCA
import torch.nn.functional as F
from glob import glob
import math

###############################################################################
# 1) Utility: Conversions between PyTorch and NumPy
###############################################################################
def torch_to_numpy(X: torch.Tensor) -> np.ndarray:
    """
    Move a PyTorch tensor to CPU and convert to a NumPy array.
    """
    return X.detach().cpu().numpy()

def numpy_to_torch(X_np: np.ndarray, device=None) -> torch.Tensor:
    """
    Convert a NumPy array to a PyTorch tensor, optionally pinning to a device (CPU/GPU).
    """
    t = torch.from_numpy(X_np)
    if device is not None:
        t = t.to(device)
    return t


###############################################################################
# 2) Single PCA (using scikit-learn) + transform / inverse_transform
###############################################################################
def pca_fit_sklearn(X: torch.Tensor, n_components: int):
    """
    Fit an sklearn PCA model on the PyTorch data X, keeping n_components.

    Returns:
        pca_model: The fitted PCA model (from sklearn).
    """
    # Convert to NumPy for scikit-learn
    X_np = torch_to_numpy(X)
    pca_model = PCA(n_components=n_components)
    pca_model.fit(X_np)
    return pca_model

def pca_transform_sklearn(X: torch.Tensor, pca_model: PCA) -> torch.Tensor:
    """
    Apply the fitted PCA transform (sklearn) to a PyTorch tensor X.
    Returns a PyTorch tensor.
    """
    X_np = torch_to_numpy(X)
    X_reduced_np = pca_model.transform(X_np)  # shape (N, n_components)
    return numpy_to_torch(X_reduced_np, device=X.device)

def pca_inverse_transform_sklearn(X_reduced: torch.Tensor, pca_model: PCA) -> torch.Tensor:
    """
    Inverse transform from reduced dimension back to original, returning a PyTorch tensor.
    """
    X_reduced_np = torch_to_numpy(X_reduced)
    X_approx_np = pca_model.inverse_transform(X_reduced_np)  # shape (N, d_original)
    return numpy_to_torch(X_approx_np, device=X_reduced.device)


###############################################################################
# 3) Blockwise (Chunked) PCA
###############################################################################
def blockwise_pca_fit_sklearn(X: torch.Tensor, block_size: int, n_components_block: int):
    """
    Split the dimension of X into chunks of size block_size and fit an sklearn PCA to each block.

    Args:
        X:                (N, d) PyTorch tensor
        block_size:       how many dimensions per chunk
        n_components_block: number of PCA components to keep in each chunk

    Returns:
        pca_models: list of fitted PCA models (one per block)
    """
    N, d = X.shape
    num_blocks = d // block_size
    pca_models = []

    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        X_block = X[:, start:end]  # (N, block_size)

        pca_model_i = pca_fit_sklearn(X_block, n_components_block)
        pca_models.append(pca_model_i)

    return pca_models

def blockwise_pca_transform_sklearn(X: torch.Tensor, pca_models: list) -> torch.Tensor:
    """
    Transform each block of X using a list of pretrained PCA models (sklearn).

    Returns a PyTorch tensor of shape (N, sum_of_blockwise_components).
    """
    N, d = X.shape
    block_size = pca_models[0].components_.shape[1]  # each PCA model was fit on block_size dims
    num_blocks = len(pca_models)

    transformed_blocks = []
    for i, pca_model_i in enumerate(pca_models):
        start = i * block_size
        end = start + block_size
        X_block = X[:, start:end]  # (N, block_size)

        X_block_reduced = pca_transform_sklearn(X_block, pca_model_i)  # (N, n_components_block)
        transformed_blocks.append(X_block_reduced)

    # Concatenate along dimension 1 (features)
    return torch.cat(transformed_blocks, dim=1)  # shape (N, num_blocks * n_components_block)

def blockwise_pca_inverse_transform_sklearn(X_reduced: torch.Tensor, pca_models: list) -> torch.Tensor:
    """
    Reconstruct each block from its reduced representation using the fitted PCA models.

    X_reduced:  (N, total_reduced_dim)
    pca_models: each model has n_components_block

    Returns: (N, original_d)
    """
    N, total_reduced_dim = X_reduced.shape
    n_components_block = pca_models[0].n_components_
    num_blocks = len(pca_models)

    reconstructed_blocks = []
    for i, pca_model_i in enumerate(pca_models):
        start_dim = i * n_components_block
        end_dim = start_dim + n_components_block

        X_block_reduced = X_reduced[:, start_dim:end_dim]  # (N, n_components_block)
        X_block_approx = pca_inverse_transform_sklearn(X_block_reduced, pca_model_i)  # (N, block_size)
        reconstructed_blocks.append(X_block_approx)

    return torch.cat(reconstructed_blocks, dim=1)  # (N, d)


###############################################################################
# 4) Optional Second-Stage PCA (again in sklearn)
###############################################################################
def second_stage_pca_fit_sklearn(X: torch.Tensor, n_components: int):
    """
    Fit a second-stage PCA on the already blockwise-reduced data X.
    """
    return pca_fit_sklearn(X, n_components)

def second_stage_pca_transform_sklearn(X: torch.Tensor, pca_model: PCA) -> torch.Tensor:
    """
    Transform with the second-stage PCA.
    """
    return pca_transform_sklearn(X, pca_model)

def second_stage_pca_inverse_transform_sklearn(X_reduced: torch.Tensor, pca_model: PCA) -> torch.Tensor:
    """
    Inverse transform from second-stage PCA.
    """
    return pca_inverse_transform_sklearn(X_reduced, pca_model)

def pad_to_chunk_multiple(x, chunk_size):
    shape = x.shape
    if len(shape)<2:
        x =x.unsqueeze(0)
        shape = x.shape
    max_in = chunk_size*math.ceil(shape[1]/chunk_size)
    if max_in> shape[1]:
        delta1 = max_in - shape[1]
        x =F.pad(x, (0, delta1, 0, 0), "constant", 0)
    return x
###############################################################################
# 5) End-to-end demonstration
###############################################################################
from glob import glob
if __name__ == "__main__":
    # Create some synthetic data in PyTorch
    # N = 1000
    # d = 1024
    # X = torch.randn(N, d)  # shape (1000, 1024)
    data = torch.load('../Datasets/llmdata/llama_3_1_8B_inst_full_block_and_ln_.pt')
    pca_weight ={}

    # 1) Fit blockwise PCA on chunks of dimension
    block_size = 262144
    n_components_block = 16384  # reduce each 256-d chunk to 64 dims
    model_dict={}
    for k, X in glob(data.items()):
        X=pad_to_chunk_multiple(X, block_size)
        X = torch.split(X, split_size_or_sections=block_size, dim=-1)
        X = torch.cat(X, dim=0).float()
        pca_models = blockwise_pca_fit_sklearn(X, block_size, n_components_block)
        model_dict[k] = pca_models

        # 2) Transform blockwise
        X_blockwise_reduced = blockwise_pca_transform_sklearn(X, pca_models)
        print("[Blockwise PCA] Reduced shape:", X_blockwise_reduced.shape)
        # => (1000, 4 * 64) = (1000, 256) if d=1024 and block_size=256
        pca_weight[k]=X_blockwise_reduced
        #
        # # 3) Optional second-stage PCA (further reduce from 256 -> 64)
        # n_components_second = 64
        # pca_model_2 = second_stage_pca_fit_sklearn(X_blockwise_reduced, n_components_second)
        # X_final = second_stage_pca_transform_sklearn(X_blockwise_reduced, pca_model_2)
        # print("[Second-Stage PCA] Final shape:", X_final.shape)
        # # => (1000, 64)
        #
        # # -----------------
        # # Reconstruction
        # # -----------------
        # # A) Inverse second-stage
        # X_blockwise_approx = second_stage_pca_inverse_transform_sklearn(X_final, pca_model_2)
        # print("[Inverse 2nd Stage] shape:", X_blockwise_approx.shape)
        # # => (1000, 256)
        #
        # # B) Inverse blockwise PCA
        # X_approx = blockwise_pca_inverse_transform_sklearn(X_blockwise_approx, pca_models)
        # print("[Inverse Blockwise] Reconstructed shape:", X_approx.shape)
        # # => (1000, 1024)
        #
        # # Evaluate MSE to see how much information was lost
        # mse = torch.mean((X - X_approx) ** 2)
        # print("[Reconstruction MSE]", mse.item())
    torch.save(model_dict, "ppdata/paca_llama3-1-8b_models.pt")
    torch.save(model_dict, "ppdata/paca_llama3-1-8b_models_data.pt")
