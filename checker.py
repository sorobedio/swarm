import os
from transformers import AutoModelForCausalLM
import torch





if __name__ == "__main__":
    model_names = ["code_alpaca", "cot", "flan_v2", "gemini_alpaca", "lima", "oasst1", "open_orca", "science", "sharegpt", "wizardlm"]
    vect_weights={}
    weights ={}
    wl =[]
    for model_name in model_names:
        model = AutoModelForCausalLM.from_pretrained("bunsenfeng/"+model_name)
        std = model.state_dict()
        for key in std.keys():
            if 'lora' in key:
                w = std[key].reshape(1,-1)
                # vect_weights[key]=w
                wl.append(w)
        weights[model_name]=torch.cat(wl, dim=-1)





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