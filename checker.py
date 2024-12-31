import os
from transformers import AutoModelForCausalLM
import torch
import random





if __name__ == "__main__":
    model_names = ["code_alpaca", "cot", "flan_v2", "gemini_alpaca", "lima", "oasst1", "open_orca", "science", "sharegpt", "wizardlm"]
    vect_weights={}
    weights ={}

    for model_name in model_names:
        model = AutoModelForCausalLM.from_pretrained("bunsenfeng/"+model_name)
        std = model.state_dict()
        wl = []
        for key in std.keys():
            if 'lora' in key:
                w = std[key].reshape(1,-1)
                # vect_weights[key]=w
                wl.append(w)
        ws= torch.cat(wl, dim=-1)
        print(f'---params:{model_name}---{ws.shape}---')
        weights[model_name]=ws


    particles_now=len(model_names)
    initial_experts_num=20
    for i in range(initial_experts_num - particles_now):
        parent_1 = random.choice(model_names)
        parent_2 = random.choice(model_names)
        while parent_1 == parent_2:
            parent_2 = random.choice(model_names)
        w_1 = random.random() * 2  # half interpolation, half extrapolation
        w_2 = 1 - w_1
        p1 = weights[parent_1]
        p2 = weights[parent_2]
        pm = w_1 * p1 + w_2 * p2
        k= parent_1+'_'+parent_2
        weights[k]=pm
    torch.save(weights, "../Datssets/llmdata/gemina_lora_weights.pt")
    print(len(weights))

    exit(0)







    # model_name = "code_alpaca"
    # model = AutoModelForCausalLM.from_pretrained("bunsenfeng/"+model_name)
    # std = model.state_dict()
    # weights =[]
    # for key in std.keys():
    #     if 'lora' in key:
    #         w = std[key].reshape(-1)
    #         print(f'---params:{key}---{w.shape}---')
    #         weights.append(w)
    # print(len(weights))
    # w = torch.cat(weights, dim=-1)
    # print(w.shape)