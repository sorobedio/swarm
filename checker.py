import os
from transformers import AutoModelForCausalLM
import torch
import random





if __name__ == "__main__":
    # wds = torch.load('wdata/utilities_vae_lora_mmlu.pt')
    # print(len(wds))
    # exit()
    model_names = ["code_alpaca", "cot", "flan_v2", "gemini_alpaca", "lima", "oasst1", "open_orca", "science", "sharegpt", "wizardlm"]
    vect_weights={}
    weights ={}
    base_model = "google/gemma-7b-it"
    random.seed(42)
    epsilon = 1e-8

    # # Generate a random float in the range (epsilon, 1 - epsilon)
    # scalar = random.uniform(epsilon, 1 - epsilon)

    for model_name in model_names:
        adapter_id="bunsenfeng/"+model_name
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)
        model.load_adapter(adapter_id)
        std = model.state_dict()
        wl = []
        for key in std.keys():
            if 'lora' in key:
                w = std[key].reshape(1,-1)
                # vect_weights[key]=w
                print(f'---params:{key}---{w.shape}--{w.dtype}-')
                wl.append(w)
        ws= torch.cat(wl, dim=-1)
        print(f'---params:{model_name}---{ws.shape}--{ws.min()}-{ws.max()}-')
        weights[model_name]=ws
    exit()
    # neweights ={}
    # weights= torch.load('wdata/reconstruct_lora_weights.pt')
    model_names = list(weights)
    particles_now=len(model_names)
    initial_experts_num=40 ##51380224
    for i in range(initial_experts_num - particles_now):
        parent_1 = random.choice(model_names)
        parent_2 = random.choice(model_names)
        while parent_1 == parent_2:
            parent_2 = random.choice(model_names)
        # w_1 = random.random() * 2  # half interpolation, half extrapolation
        w_1 = random.uniform(epsilon, 1 - epsilon)
        w_2 = 1 - w_1
        p1 = weights[parent_1]
        p2 = weights[parent_2]
        pm = w_1 * p1 + w_2 * p2
        k= parent_1 +'_and_'+ parent_2
        weights[k]=pm
        # neweights[k]=pm
    torch.save(weights, "../Datasets/llmdata/gemina7b_it_lora_weights_recon_ext.pt")
    print(len(neweights))
    exit(0)







