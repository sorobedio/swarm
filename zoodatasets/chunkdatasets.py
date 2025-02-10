import os

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
from torch.utils.data import Dataset
from glob import glob
import math

def load_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


# Logarithmic Transform (for heavy-tailed data)
def log_transform(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))

# Inverse Logarithmic Transform (recover original values)
def inverse_log_transform(x_transformed):
    return torch.sign(x_transformed) * (torch.expm1(torch.abs(x_transformed)))

def arsh_transform(x):
    """Applies the ArcSinh (ArSh) transformation to expand small values smoothly."""
    return torch.asinh(x)

def inverse_arsh_transform(x_transformed):
    """Inverse of the ArcSinh (ArSh) transformation."""
    return torch.sinh(x_transformed)


# arsh_transform = transforms.Lambda(lambda x: torch.asinh(x))

def matpadder(x, max_in=512):
    shape = x.shape
    # delta1 = max_in - shape[0]
    if len(shape)<2:
        x =x.unsqueeze(0)
        shape = x.shape
    delta2 = max_in - shape[1]

    out = F.pad(x, (0, delta2, 0, 0), "constant", 0)
    return out
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
class ZooDataset(Dataset):
    """weights dataset."""
    def __init__(self, root='zoodata', dataset="joint", split='train', topk=None, scale=1.0, transform=True, normalize=False,
                 max_len=262144):
        super(ZooDataset, self).__init__()
        #1960513  3145728   25165824
        self.topk = topk
        self.max_len = max_len
        self.split = split
        self.dataset = dataset
        self.normalize = normalize
        self.chunk_size = max_len
        self.scale=scale
        # datapath = os.path.join("../Datasets", f'llmdata/llama_3_2_1B_inst_full_block_and_ln.pt')
        # datapath = os.path.join("../Datasets", f'llmdata/llama_3b_mlp_.pt')
        # datapath = os.path.join("../Datasets", f'llmdata/llama_3_3b_full_.pt')
        # datapath = os.path.join(root, f'llmdata/llama_3b_self_attn_.pt')  # 262144
        # datapath = os.path.join("../Datasets", f'llmdata/llama_3_8b_self_attn_.pt')
        datapath = os.path.join("../Datasets", f'llmdata/llama_3_8b_full_.pt')
        #'../Datasets/llmdata/llama_3_8b_self_attn_.pt'
        #'../Datasets/llmdata/llama_3b_self_attn_.pt'
        #../Datasets/llmdata/llama_8b_mlp_.pt ../Datasets/llmdata/llama_3b_mlp_.pt
        # self.transform =  transforms.Lambda(lambda x: torch.asinh(x))
        data= self.load_data(datapath, dataset=dataset)

        print(f'======{data.dtype}=========dataset size=={data.shape}======max={data.max()}======={data.min()}==========')
        # data = 2*(data-data.min())/(data.max()-data.min())-1

        self.data = data.cpu()
        print(f'===============dataset size=={data.shape}======max={data.max()}======={data.min()}==========')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        weight = self.data[idx]/self.scale
        # if self.transform:
        # weight = self.transform(weight)
        sample= {'weight':weight, 'dataset': []}
        return  sample
    def load_data(self, file, dataset='joint'):
        data = torch.load(file)
        wl = []
        if dataset=='joint':
            keys = list(data)
            # keys.remove('layernorm.weight')
            # keys = ['sharegpt_cot', 'gemini_alpaca_sharegpt']
            # keys =keys[:1]
            # print(keys)

            for k in keys:
                w = data[k]
                print(w.shape, w.dtype)
                w=pad_to_chunk_multiple(w, chunk_size=self.chunk_size)
                if self.normalize == "z_score":
                    u = torch.mean(w, dim=1)
                    v = torch.std(w, dim=1)
                    w = (w - u[:, None]) / v[:, None]
                elif self.normalize == "min_max":
                    x_max, _ = torch.max(w, dim=-1)
                    x_min, _ = torch.min(w, dim=-1)
                    xdiff = x_max - x_min
                    w = 2 * (w - x_min[:, None]) / xdiff[:, None] - 1
                w = torch.split(w, split_size_or_sections=self.chunk_size, dim=-1)
                w = torch.cat(w, dim=0)

                if self.topk is not None:
                    if self.topk > 0:
                        w = w[:self.topk]
                        wl.append(w)
                else:
                    wl.append(w)
            data = torch.cat(wl, dim=0)
        else:
            w = data[dataset]
            w = pad_to_chunk_multiple(w, chunk_size=self.chunk_size)
            w = torch.split(w, split_size_or_sections=self.chunk_size, dim=-1)
            w = torch.cat(w, dim=0)
            if self.normalize == "z_score":
                u = torch.mean(w, dim=1)
                v = torch.std(w, dim=1)
                w = (w - u[:, None]) / v[:, None]
            elif self.normalize == "min_max":
                x_max, _ = torch.max(w, dim=-1)
                x_min, _ = torch.min(w, dim=-1)
                xdiff = x_max - x_min
                w = (w - x_min[:, None]) / xdiff[:, None]
            if self.topk is not None:
                if self.topk > 0:
                    w = w[:self.topk]

            data = w
        return data