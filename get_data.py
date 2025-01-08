import os
import numpy as np
import torch
import torch.nn as nn
from transformers import GPTJForCausalLM
import torch

from transformers import BertTokenizer, BertModel

# from quality_check import model_id


def get_weights(std):
    ws = []
    for p in std:
        if "num_batches_tracked" in p:
            continue
        # if  'running_mean' or 'running_var' in p:
        #     continue
        else:
            w = std[p].detach().cpu().reshape(-1)
            # if len(w)>0:
            ws.append(w)
    ws = torch.cat(ws, dim=-1)
    return ws

def gets_weights(std):
    # std = model.state_dict()
    weights = []
    for params in std:
        if not params.endswith('num_batches_tracked'):
            if 'mean' in params or 'var' in params:
                continue
            elif 'rotary_emb' in params:
                print(f'found------{params}-------------')
                continue
            # print(params)
            print(f'paramertes================={params}---------------------')
            w = std[params].reshape(-1)
            weights.append(w)
    return torch.cat(weights, -1)

def extract_layers_weights(std):
    # std = model.state_dict()
    weights = {}
    ws = []
    for params in std:
        if not params.endswith('num_batches_tracked'):
            if 'mean' in params or 'var' in params:
                continue

            w = std[params].reshape(1,-1)
            print(f'paramertes============={params}---------------------')
            print(w.shape)
            print(w.min(), w.max())
            ws.append(w)
            weights[str(params)]=w
    return weights, torch.cat(ws, dim=-1)

def extract_layer_weights(std, tgt='norm', pref=None):
    # std = model.state_dict()
    weights = {}
    ws = []
    for params in std:
        if not params.endswith('num_batches_tracked'):
            if 'mean' in params or 'var' in params:
                continue
            # print(params)
            if tgt in params:
                w = std[params].reshape(1,-1)
                print(f'paramertes============={params}---------------------')
                print(w.shape)
                print(w.min(), w.max())
                ws.append(w)
                if pref is not None:
                    key = f'{pref}_{str(params)}'
                # weights[key]=w
                else:
                    key = f'{str(params)}'
                weights[key] = w
    return weights, ws


def extract_layer_weights_withexc(std, tgt='norm', pref=None):
    # std = model.state_dict()
    weights = {}
    ws = []
    for params in std:
        if not params.endswith('num_batches_tracked'):
            if 'mean' in params or 'var' in params:
                continue
            if "norm" in params:
                continue
            # print(params)
            if tgt in params:
                continue
            w = std[params].reshape(1,-1)
            print(f'paramertes============={params}---------------------')
            print(w.shape)
            print(w.min(), w.max())
            ws.append(w)
            if pref is not None:
                key = f'{pref}_{str(params)}'
            # weights[key]=w
            else:
                key = f'{str(params)}'
            weights[key] = w
    return weights, ws



def extract_tflayer_weights(std, cond='norm', tag='layer', pref=None):
    # std = model.state_dict()
    weights = {}
    ws = []
    for params in std:
        if 'layer' in params:
            if tgt in params:
                w = std[params].reshape(1,-1)
                print(f'paramertes============={params}---------------------')
                print(w.shape)
                print(w.min(), w.max())
                ws.append(w)
                if pref is not None:
                    key = f'{pref}_{str(params)}'
                # weights[key]=w
                else:
                    key = f'{str(params)}'
                weights[key] = w
    return weights, ws


def extract_layer_weights_with_b(std, tgt='norm'):
    # std = model.state_dict()
    weights = {}
    ws = []
    for params in std:
        if not params.endswith('num_batches_tracked'):
            if 'mean' in params or 'var' in params:
                continue
            # print(params)
            if tgt in params:
                if 'bias' in params:
                    continue
                w = std[params].reshape(1,-1)
                p = params.replace('weight', 'bias')
                try:
                    b = std[p].reshape(1, -1)
                    print(f'paramertes============={params}-------{w.shape}------{b.shape}--------')
                    w = torch.cat((w, b), dim=-1)
                except:
                    print(f'paramertes============={params}-------{w.shape}------no bias--------')

                print(w.shape)
                print(w.min(), w.max())
                ws.append(w)
                weights[str(params)]=w
    return weights, ws


def extract_layers_with_b(std, tgt='norm'):
    # std = model.state_dict()
    weights = {}
    ws = []
    for params in std:
        if not params.endswith('num_batches_tracked'):
            if 'mean' in params or 'var' in params:
                continue
            # print(params)
            if tgt not in params:
                if 'bias' in params:
                    continue
                w = std[params].reshape(1, -1)
                p = params.replace('weight', 'bias')
                try:
                    b = std[p].reshape(1, -1)
                    print(f'paramertes============={params}-------{w.shape}------{b.shape}--------')
                    w = torch.cat((w, b), dim=-1)
                except:
                    print(f'paramertes============={params}-------{w.shape}------no bias--------')

                print(w.shape)
                print(w.min(), w.max())
                ws.append(w)
                weights[str(params)] = w
    return weights, ws

def get_layer_weights(std, tgt='norm'):
    # std = model.state_dict()
    weights = []
    for params in std:
        if not params.endswith('num_batches_tracked'):
            if 'mean' in params or 'var' in params:
                continue
            # print(params)
            if tgt in params:
                w = std[params].reshape(-1)
                weights.append(w)
    return torch.cat(weights, -1)
# def get_
#
def extract_weights(std):
    # std = model.state_dict()
    weights = []
    for params in std:
        if not params.endswith('num_batches_tracked'):
            if 'mean' in params or 'var' in params:
                continue
            elif 'rotary_emb' in params:
                continue
            print(f'paramertes================={params}---------------------')
            w = std[params]
            weights.append(w)
    return torch.stack(weights, 0)


def get_blocks_weights(std, tgt='norm', cond='layer'):
    # std = model.state_dict()
    weights = {}
    ws = []
    for params in std:
        if not params.endswith('num_batches_tracked'):
            if 'mean' in params or 'var' in params:
                continue
            elif 'rotary_emb' in params:
                continue
            # print(params)
            if tgt not in params and cond in params:
                w = std[params].reshape(1,-1)
                print(f'paramertes============={params}---------------------')
                print(w.shape)
                print(w.min(), w.max())
                ws.append(w)
                weights[str(params)]=w
    return weights, ws


def extract_mlp_weights_with_bias(state_dict):
    # Load the model
    # model = AutoModel.from_pretrained(model_name)

    # Extract the state dictionary
    # Dictionary to store MLP weights (with bias if present) in vectorized form
    mlp_weights_vectorized = {}

    # Identify and process MLP layers only
    for name, param in state_dict.items():
        if 'mlp' in name and 'weight' in name:  # Ensuring it’s an MLP weight layer
            # Get the base layer name by removing the '.weight' suffix
            base_name = name.rsplit('.', 1)[0]
            bias_name = f"{base_name}.bias"

            # Flatten the weight tensor
            weight_vector = param.flatten()

            # Check if there's a corresponding bias and concatenate if present
            if bias_name in state_dict:
                bias_vector = state_dict[bias_name].flatten()
                combined_vector = torch.cat([weight_vector, bias_vector])
            else:
                combined_vector = weight_vector

            # Store the combined vector in the dictionary
            print(combined_vector.shape)
            mlp_weights_vectorized[name] = combined_vector

    return mlp_weights_vectorized


def set_mlp_weights(model, mlp_weights_vectorized=None):
    # Load the model
    # model = AutoModel.from_pretrained(model_name)
    # Check if mlp_weights_vectorized is provided
    if mlp_weights_vectorized is None:
        raise ValueError("Please provide a dictionary of MLP weights in vectorized form.")
    # Load the model's current state dictionary
    state_dict = model.state_dict()
    # Iterate over the provided MLP weights
    for name, vector in mlp_weights_vectorized.items():
        # Retrieve the weight tensor shape from the model's original state dictionary
        if name in state_dict:
            original_shape = state_dict[name].shape
            weight_size = state_dict[name].numel()

            # Split the vector into weight and bias if needed
            if vector.numel() > weight_size:
                weight_vector = vector[:weight_size]
                bias_vector = vector[weight_size:]

                # Reshape and set the weight and bias back to the model's state dictionary
                state_dict[name].copy_(weight_vector.view(original_shape))
                state_dict[name.replace("weight", "bias")].copy_(bias_vector)
            else:
                # Reshape and set the weight tensor
                state_dict[name].copy_(vector.view(original_shape))

    # Load the modified state dictionary back into the model
    model.load_state_dict(state_dict)

    return model


# from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Tokenizer, GPT2Model
import torch
import transformers
import yaml
from transformers import GPT2Tokenizer, GPT2Model
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import GPTNeoXForCausalLM, AutoTokenizer
# from sklearn.decomposition import PCA, IncrementalPCA

#python ~/miniconda3/envs/lm3/lib/python3.8/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir ./llama-2-7b/ --model_size 7B --output_dir meta-llama


# python ~/miniconda3/envs/lm3/lib/python3.8/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir  ./Meta-Llama-3-8B/ --model_size  8B --output_dir ./Meta-Llama-3-8B-hf

 #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
 #https://ai.meta.com/blog/5-steps-to-getting-started-with-llama-2/
#!huggingface-cli login --token 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 download meta-llama/Llama-2-7b-hf --include "original/*" --local-dir meta-llama/Llama-2-7b-hf
# from transformers import AutoTokenizer, AutoConfig
# from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
#/home/user/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9
if __name__=='__main__':
    # ws = torch.load('../../Datasets/llama3_weights/best_norm_weights.pt')
    # w = ws["DeepMount00/Llama-3-8b-Ita"]
    # print(w.shape)
    # exit()
    # huggingface-cli download meta-llama/Llama-2-7b-hf --include "original/*" --local-dir meta-llama/Llama-2-7b-hf

#python convert_llama_weights_to_hf.py --input_dir ./Meta-Llama-3-8B/ --model_size  8B --output_dir ./Meta-Llama-3-8B-hf
    # modellist = ["DeepMount00/Llama-3-8b-Ita", "MaziyarPanahi/Llama-3-8B-Instruct-v0.9", "MaziyarPanahi/Llama-3-8B-Instruct-v0.8"]
    # modellist =["meta-llama/Meta-Llama-3-8B-Instruct"]
    # modellist = ["meta-llama/Meta-Llama-3-8B-Instruct", 'meta-llama/Meta-Llama-3-8B']
    # modellist = ['MaziyarPanahi/Phi-3-mini-4k-instruct-v0.3', "microsoft/Phi-3-mini-4k-instruct",
    #               "microsoft/Phi-3-mini-128k-instruct",
    #               "microsoft/Phi-3-mini-4k-instruct"]
    # modellist =  ["swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"]
    # modellist = ["microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3-mini-128k-instruct", ]
    # modellist = ["meta-llama/Meta-Llama-3.1-8B-Instruct", "VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct"]
    # modellist = ["meta-llama/Meta-Llama-3.1-8B-Instruct"]
    # modellist = ["meta-llama/Meta-Llama-3.1-70B-Instruct"]
    # modellist = ["TinyLlama/TinyLlama_v1.1", "TinyLlama/TinyLlama_v1.1_math_code"]

    # modellist = ["VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct", "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2"]

    # model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # modellist =["microsoft/Phi-3-medium-128k-instruct"]
    # modellist = ["microsoft/Phi-3-medium-4k-instruct"]
    # modellist = ["microsoft/Phi-3-small-8k-instruct"]
    # modellist = ["microsoft/Phi-3-small-8k-instruct"]
    # modellist=["arcee-ai/Llama-3.1-SuperNova-Lite"]
    # modellist = ["mistralai/Mistral-7B-Instruct-v0.3"]
    # modellist = ["meta-llama/Meta-Llama-3.1-8B-Instruct"]
#python spectrum.py --model-name mistralai/Mistral-7B-Instruct-v0.3 --top-percent 25
    #meta-llama/Llama-3.2-3B
    # modellist=["meta-llama/Llama-3.2-1B-Instruct"]
    # modellist = ["meta-llama/Llama-3.2-3B-Instruct"]
    # modellist = ["meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"]
    # modellist = ["meta-llama/Llama-3.2-3B"]
    #

    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # # model = GPT2Model.from_pretrained('gpt2')
    # model = GPT2LMHeadModel(config)
    # # print(model.h[0].attn)
    # # exit()
    #
    # modellist = ["HuggingFaceTB/SmolLM2-135M-Instruct", 'HuggingFaceTB/SmolLM2-135M' ]
    # modellist=['google/gemma-7b-it']
    # modellist = ["meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"]

    # modellist = ["EleutherAI/pythia-410m"]
    # modellist = ["facebook/MobileLLM-125M","facebook/MobileLLM-350M", "facebook/MobileLLM-600M"]
    # modellist = ["facebook/MobileLLM-125M"]


    # model = GPTNeoXForCausalLM.from_pretrained(
    #     "EleutherAI/pythia-70m-deduped",
    #     revision="step143000",
    # )
    modellist=["meta-llama/Meta-Llama-3.1-8B-Instruct"]


    #


    wl = []
    x_min = -0.3398
    x_max = 0.3574
    weights = {}
    mw ={}
    ws =[]

    # model = GPTNeoXForCausalLM.from_pretrained(
    #     "EleutherAI/pythia-160m-deduped",
    #     revision="step143000",
    #     cache_dir="./pythia-160m-deduped/step143000", torch_dtype=torch.bfloat16
    # )
    i=0
    for md in modellist:
        print(f'processing-----{md}----------')
        # model = LlamaForCausalLM.from_pretrained(md)
        model = AutoModelForCausalLM.from_pretrained(
                md,
                device_map="cpu",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            # revision="step143000",
            )
        # print(model)
        # exit()
        std = model.state_dict()
        # k=str(md.split("/")[-1])
        # w, we = extract_layer_weights(std, tgt='norm', pref=k)
        # we = get_layer_weights(std, tgt='norm')
        w, we = extract_layer_weights_withexc(std, tgt='layer', pref=None)
        # w, we = get_blocks_weights(std, tgt='norm', cond='layer')

        # we = gets_weights(std)
        # print(we.shape, we.min(), we.max(), we.dtype)
        # exit()
        weights.update(we) #67584
        # weights[k] = we
    torch.save(weights, '../Datasets/llmdata/llama-3-1-8b_layer_norm.pt')  # 1498482688
    print(len(weights))
    exit()
    #HuggingFaceTB/SmolLM2-135M-Instruct
    #'../Datasets/llmdata/llama_3_2_1b_3b_inst_selft_atten__.pt'
    # '../Datasets/llmdata/llama_3_2_1b_3b_inst_mlp_.pt'
#embed_tokens.weight
#3606752256
#1498482688==128x11706896==1463362 1024