import os
import numpy as np
import torch
import torch.nn as nn
# from transformers import GPTJForCausalLM
import torch

# from transformers import BertTokenizer, BertModel

# from mdt.modules.infer_mdt import device


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
            # if pref is not None:
            #     key = f'{pref}_{params}'
            # else:
            #     key = f'{params}'
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

def extract_layer_weights(std, tgt='norm'):
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
                weights[str(params)]=w
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


def concatenate_tensors(dict_list):
    """
    Concatenates tensor values from a list of dictionaries with the same keys.

    Parameters:
        dict_list (list): List of dictionaries where keys are the same across all dictionaries
                          and values are PyTorch tensors.

    Returns:
        dict: A single dictionary where each key maps to the concatenated tensor values.
    """
    if not dict_list:
        return {}

    # Initialize an empty dictionary to store concatenated tensors
    concatenated_dict = {}

    # Iterate over keys in the first dictionary
    for key in dict_list[0].keys():
        # Concatenate tensors along the first dimension (default behavior of torch.cat)
        concatenated_dict[key] = torch.stack([d[key] for d in dict_list], dim=0)

    return concatenated_dict

# from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Tokenizer, GPT2Model
import torch
import transformers
import yaml
from transformers import GPT2Tokenizer, GPT2Model
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import GPTNeoXForCausalLM, AutoTokenizer

#python ~/miniconda3/envs/lm3/lib/python3.8/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir ./llama-2-7b/ --model_size 7B --output_dir meta-llama


# python ~/miniconda3/envs/lm3/lib/python3.8/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir  ./Meta-Llama-3-8B/ --model_size  8B --output_dir ./Meta-Llama-3-8B-hf

 #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
 #https://ai.meta.com/blog/5-steps-to-getting-started-with-llama-2/
#!huggingface-cli login --token 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 download meta-llama/Llama-2-7b-hf --include "original/*" --local-dir meta-llama/Llama-2-7b-hf
# from transformers import AutoTokenizer, AutoConfig
# from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
#/home/user/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9
if __name__=='__main__':

    model_id= "EleutherAI/pythia-160m"
    # model_id = "EleutherAI/pythia-410m-deduped"
    start_step = 1000
    end_step = 143000
    step_size=1000
    weights = []
    weights_dict= {}
    mw ={}
    # for i in [13000, 39000, 65000, 91000, 117000, 143000]:
    for i in range(start_step, end_step+1, step_size):
        print(f'processing-----{i}----------')
        # model = LlamaForCausalLM.from_pretrained(md)
        model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="cpu",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            revision=f"step{str(i)}",
            )

        std = model.state_dict()
        w = gets_weights(std)  # extract full model weights
    #     # w, we = get_blocks_weights(std, tgt='norm', cond='mlp')
    #     w, we = extract_layer_weights_with_b(std, tgt='mlp')
    #     # w, we = extract_layers_with_b(std, tgt='ln_')
        weights.append(w)

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     # device_map=device,
    #     # torch_dtype=torch.bfloat16,
    #     trust_remote_code=True,
    #     # revision=f"step{str(i)}",
    # )
    # std = model.state_dict()
    k = model_id.split('/')[-1]
    weights = torch.stack(weights, 0)
    mw[k] = weights
    print(weights.shape)
    # w, we = extract_layer_weights_with_b(std, tgt='mlp')
    # weights_dict.update(w)
    # weights= concatenate_tensors(weights)

    torch.save(mw, f'../Datasets/modelszoo/pythia_160m_full_13000_by_143000_b16_.pt')
    # torch.save(mw, f'../Datasets/llmdata/pythia-70m-{start_step}_{end_step}.pt')
    # torch.save(mw, '../Datasets/70mzoo/full_pythia_70_.pt')  # 1498482688
    # print(w.shape, w.min(), w.max())
    # exit()

