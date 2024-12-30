import os
from transformers import AutoModelForCausalLM
import torch

model_name = "code_alpaca"
model = AutoModelForCausalLM.from_pretrained("bunsenfeng/"+model_name)
std = model.state_dict()
weights =[]
for key in std.keys():
    if 'lora' in key:
        weights.append(std[key].reshape(1, -1))
print(len(weights))
w = torch.cat(weights, dim=-1)
print(w.shape)