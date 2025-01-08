
import os

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset

def load_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def matpadder(x, max_in=512):
    shape =x.shape
    # delta1 = max_in - shape[0]
    delta2 = max_in - shape[1]

    out = F.pad(x, (0, delta2, 0, 0), "constant", 0)
    return out

class ZooDataset(Dataset):
    """weights dataset."""
    def __init__(self, root='zoodata', dataset='joint', split='train', scale=1.0, topk=None, transform=None, normalize=False,
                 max_len=4096):
        super(ZooDataset, self).__init__()
        self.dset = dataset
        self.topk=topk
        self.max_len = max_len
        self.normalize = normalize
        self.split=split
        self.scale= scale

        # datapath = os.path.join(root, f'llama3_weights/llama_3_8_1_norm_dict_instruct_.pt')
        # datapath = os.path.join(root, f'llama_weights/base_mistral_7b_inst_ln_.pt')

        # datapath = os.path.join(root, f'llmdata/llama_3_2_1b_3b_inst_norm__.pt')

        # datapath = os.path.join(root, f'llmdata/gemina7b_it_lora_weights.pt')
        # '../../Datasets/llama_weights/
        # "../Datasets/llmdata/gemina7b_it_lora_weights.pt"

        datapath = os.path.join(root, f'llmdata/llama-3-1-8b_layer_norm.pt')
        #llmdata/llama-3-1-8b_layer_norm.pt


        self.transform = transform
        data = self.load_data(datapath, dataset)
        # x_min, x_max = data.min(), data.max()
        print('===============dataset size=========================')
        # print(self.data.shape, x_min, x_max)
        print(f'===============dataset size=={data.shape}======max={data.max()}======={data.min()}==========')
        # exit()
        print('========================================')
        # data = 2 * (data - x_min) / (x_max - x_min) - 1
        self.data = data/self.scale

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        weight = self.data[idx]
        if self.transform:
            weight = self.transform(weight)
        sample = {'weight': weight, 'dataset': []}
        return  sample
    def load_data(self, file, dset='joint'): #loading pretrained weights vector from dict {datset:weights)
        data = torch.load(file)
        if dset=='joint':
            wl = []
            keys = list(data)
            for k in keys:
                w = data[k].detach().cpu()
                # print(w.shape)
                # w = w.reshape(1, -1)
                if len(w.shape)<2:
                    w = w.unsqueeze(0)
                if w.shape[-1] < self.max_len:
                    w = matpadder(w, self.max_len)
                if self.normalize == 'min_max':
                    x_max = w.max()
                    x_min = w.min()
                    w = 2*(w - x_min) / (x_max - x_min)-1
                elif self.normalize == 'z_score':
                    u = torch.mean(w)
                    var = torch.std(w)
                    w = (w - u) / var
                if self.topk is not None:
                    # print(w)
                    wl.append(w[:self.topk])
                else:
                    wl.append(w)
            datas = torch.cat(wl, dim=0)
        else:
            ws =None
            w = data[dset].detach().cpu()
            w = w.reshape(1, -1)
            if len(w.shape) < 2:
                w = w.unsqueeze(0)
            if w.shape[-1] < self.max_len:
                w = matpadder(w, self.max_len)
            if self.normalize=='min_max':
                x_max = w.max()
                x_min = w.min()
                w = 2*(w - x_min) / (x_max - x_min)-1
            elif self.normalize == 'z_score':
                u = torch.mean(w)
                var = torch.std(w)
                w = (w-u)/var
            if self.topk is not None:
               ws =w[:self.topk]
            else:
                ws = w

            datas=ws
        return datas


