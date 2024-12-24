import os

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset
from glob import glob
import math

def load_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

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
    def __init__(self, root='zoodata', dataset="joint", split='train', topk=None, scale=1.0, transform=None, normalize=False,
                 max_len=1048576):
        super(ZooDataset, self).__init__()
        #128256 25165824
        self.topk = topk
        self.max_len = max_len
        self.split = split
        self.dataset = dataset
        self.normalize = normalize
        self.chunk_size = max_len
        self.scale=scale

        # datapath = os.path.join(root, f'llama_weights/base_llama8b_top_8_spec_.pt')
        # datapath = os.path.join(root, f'llama_weights/base_llama8b_top_8_spec_top_1.pt')
        # datapath = os.path.join(root, f'llama_weights/base_llama8b_top_8_spec_ffn.pt')
        # datapath = os.path.join(root, f'llama_weights/base_llama8b_top_8_spec_attn.pt')
        #'../Datasets/llama_weights/base_llama8b_top_8_spec_.pt'
        # llama_weights / base_llama8b_fnn_30_.pt

        # datapath = os.path.join(root, f'llama_weights/base_llama8b_fnn_30_.pt')

        # ../Datasets/llama_weights/base_llama8b_8_spec_top_25_down_proj_.pt
        # ../Datasets/llama_weights/base_llama8b_8_spec_top_25_gate_proj_.pt
        # ../Datasets/llama_weights/base_llama8b_8_spec_top_25_up_proj_.pt
        # datapath = os.path.join(root, f'llama_weights/base_llama8b_8_spec_top_25_down_proj_.pt')
        # datapath = os.path.join(root, f'llama_weights/base_llama8b_8_spec_top_25_gate_proj_.pt')
        datapath = os.path.join(root, f'llama_weights/base_llama8b_8_spec_top_25_up_proj_.pt')


        #/
        #
#
        #
        self.transform = transform
        data= self.load_data(datapath, dataset=dataset)
        self.data = data.detach().cpu()
        print('===============dataset size=========================')
        print(self.data.shape)
        print('========================================')
        # exit()
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        weight = self.data[idx].to(torch.bfloat16)
        if self.transform:
            weight = self.transform(weight)

        weight= weight/self.scale
        sample = {'weight': weight, 'dataset': []}
        return sample

    def load_data(self, file, dataset='joint'):
        data = torch.load(file)
        wl = []
        if dataset=='joint':
            keys = list(data)

            for k in keys:
                w = data[k]
                # print(w.shape)
                w=pad_to_chunk_multiple(w, chunk_size=self.chunk_size)
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
                    w = 2*(w - x_min[:, None]) / xdiff[:, None]-1
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


