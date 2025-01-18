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
        #1960513
        self.topk = topk
        self.max_len = max_len
        self.split = split
        self.dataset = dataset
        self.normalize = normalize
        self.chunk_size = max_len
        self.scale=scale
#51380224==4096 x 12544
        #'../Datasets/facebook/mobilellm_125_mlp__.pt'
        #'../Datasets/modelszoo/pythia_160m_mlp_final.pt'
        # datapath = os.path.join(root, f'modelszoo/pythia_160m_mlp_final.pt') #2362368
        # datapath = os.path.join(root, f'llmdata/SmolLM2-135M_full.pt')  # 8141328
        # datapath = os.path.join(root, f'llmdata/SmolLM2-heads_.pt')  # 2362368
        #'../Datasets/llmdata/SmolLM2-heads_.pt'
        # 11004164
        # datapath = os.path.join(root, f'modelszoo/pythia_160m_mlp_100000_143000.pt')2362368
        # datapath = os.path.join(root, f'llmdata/llama_head_.pt')#1026048
        #'../Datasets/modelszoo/pythia_410m_full_13000_by26_143000.pt'
        # datapath = os.path.join(root, f'modelszoo/pythia_410m_full_13000_by26_143000.pt')  # 2156032
        # datapath = os.path.join(root, f'modelszoo/pythia_160m_full_13000_by_143000_b16_.pt')  #
        # datapath = os.path.join(root, f'llmdata/llama-3-1-8b_layer_full.pt')  #458752
        # '../Datasets/llmdata/llama-3-1-8b_layer_full.pt'

        # datapath = os.path.join(root, f'gemina7b_it_lora_weights.pt')


        # datapath = os.path.join(root, f'llmdata/gemini_7b_int_top_25p_attn_.pt') #12582912
        #f'../Datasets/llmdata/gemini_7b_int_top_25p_attn_.pt'

        #'../Datasets/llmdata/llama_3_1_8B_inst_full_block_and_ln_.pt'

        datapath = os.path.join(root, f'llmdata/llama_3_1_8B_inst_full_block_and_ln_.pt')#262144

        # datapath = os.path.join(root, f'llmdata/llama_3_2_3B_inst_full_block_and_ln_.pt')  # 262144
        #'../Datasets/llmdata/llama_3_2_3B_inst_full_block_and_ln_.pt'



        # '../Datasets/llmdata/gemina7b_it_lora_weights.pt'
        #pythia_160m_full_13000_by_143000_b16_.pt'

        # datapath = os.path.join(root, f'modelszoo/pythia_410m_full_100000_143000.pt')  # 4401664
        #'../Datasets/modelszoo/pythia_410m_full_100000_143000.pt'
        # datapath = os.path.join(root, f'llmdata/pythia-70m-100000_143000.pt')11004164
        self.transform = transform
        data= self.load_data(datapath, dataset=dataset)
        # x_min, x_max = data.min(), data.max()

         # == == == == == == == = 9.73654277913738e-06
        std = 0.013931703753769398
        mu=9.73654277913738e-06
        # x_max = 0.9140625
        # x_min = 1.4140625

        x_min = -0.9140625
        x_max =1.4140625
        data = (data - mu) / std
        print(f'===============dataset size=={data.shape}======max={data.max()}======={data.min()}==========')
        # print(data[0][:20])
        # data = 2 * (data - x_min) / (x_max - x_min) - 1
        # mu = data.mean()
        # std = data.std()
        print('===============dataset size=========================')
        # print(self.data.shape, x_min, x_max)

        # print(f'============{std}==============={mu}=============')
        # data = (data-mu)/std

        # exit()
        self.data = data.detach().cpu()
        print(f'===============dataset size=={data.shape}======max={data.max()}======={data.min()}==========')


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
            # keys.remove('layernorm.weight')
            # keys = ['sharegpt_cot', 'gemini_alpaca_sharegpt']
            keys =keys[:14]
            # print(keys)

            for k in keys:
                w = data[k]
                print(w.shape)
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
                w = torch.cat(w, dim=0).float()

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


