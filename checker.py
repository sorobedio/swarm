import os
from transformers import AutoModelForCausalLM
import torch

for model_name in ["code_alpaca", "cot", "flan_v2", "gemini_alpaca", "lima", "oasst1", "open_orca", "science", "sharegpt", "wizardlm"]:
    if os.path.exists(model_name):
        continue
    model = AutoModelForCausalLM.from_pretrained("bunsenfeng/"+model_name)
    print(model)
    exit()
    model.save_pretrained(model_name)

model_name = "code_alpaca"
model = AutoModelForCausalLM.from_pretrained("bunsenfeng/"+model_name)
std = model.state_dict()
weights =[]
for key in std.keys():
    if 'lora' in key:
        w = std[key].reshape(-1)
        print(f'---params:{key}---{w.shape}---')
        weights.append(w)
print(len(weights))
w = torch.cat(weights, dim=-1)
print(w.shape)