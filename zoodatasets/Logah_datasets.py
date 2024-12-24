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
                 max_len= 5022410):
        super(ZooDataset, self).__init__()
        #128256 25165824 #537134  1205900 200984
        self.topk = topk
        self.max_len = max_len
        self.split = split
        self.dataset = dataset
        self.normalize = normalize
        self.chunk_size = max_len
        self.scale=scale #278325
        # datapath = os.path.join(root, f'zoodata/cifar_models_weights.pt') #best_mlp_out_norm.pt'
        # datapath = os.path.join(root, f'zoodata/pytorch_cifar_pretrained_weights.pth') #269425

        # datapath = os.path.join(root, f'zoodata/pytorch_cifar_pretrained_weights.pth')  # 269425
        # datapath = os.path.join(root, f'llama_weights/full_llama_3_2_1B_.pt')  # 731680
        #../../Datasets/llama_weights/full_llama_3_2_1B_.pt
        datapath = os.path.join(root, f'vit_cifar_10_100_logah_zoo_large.pt')
        #1498482688
        # datapath = os.path.join(root, f'zoodata/mixed_train_set_weights.pt')


        # '../../Datasets/phi_3_weights/gem_instruct_lm_head_.pt'
        # datapath = os.path.join(root, f'llama3_weights/Llama-3-1-8B-norm_.pt')
        # datapath = os.path.join(root, f'llama3_weights/tiny_llama_joint.pt')  # 537134
#../.Phi-3-mini-4k-instruct_norm_.pt'
        #
        self.transform = transform
        data= self.load_data(datapath, dataset=dataset)
        self.data = data.detach().cpu()
        print('===============dataset size=========================')
        print(self.data.shape, data.min(), data.max())
        print('========================================')
        # exit()
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        weight = self.data[idx]
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
            # keys =  torch.load('best_md.pt')

            for k in keys:
                w = data[k].detach().cpu()
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


