import os
import numpy as np
import torch
import torch.nn as nn
from transformers import GPTJForCausalLM
import torch

from transformers import BertTokenizer, BertModel
import yaml

from transformers import LlamaForCausalLM, LlamaTokenizer


def load_yaml_as_dict(yaml_content):
    """
    Load a YAML content as a dictionary.

    Parameters:
        yaml_content (str): The YAML content in string format.

    Returns:
        dict: A dictionary containing the parsed YAML structure.
    """
    try:
        # Use yaml.safe_load to parse the YAML content
        yaml_dict = yaml.safe_load(yaml_content)
        return yaml_dict
    except yaml.YAMLError as exc:
        print(f"Error loading YAML: {exc}")
        return None

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

def get_weights_from_list(std, layers):
    # std = model.state_dict()
    weights = {}
    ws = []
    for params in std:
        for layer in layers:
            if layer in params:
                w = std[params].reshape(1,-1)
                print(f'paramertes============={params}---------------------')
                print(w.shape)
                print(w.min(), w.max())
                ws.append(w)
                weights[str(params)]=w
    return weights, w


# from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Tokenizer, GPT2Model
import torch
import transformers
import yaml

from transformers import LlamaForCausalLM, LlamaTokenizer


if __name__=='__main__':

    # yaml_file = 'spectrum/layer_selection.yaml'
    # yaml_file = 'spectrum/mistral_7b_config.yaml'
    # yaml_file =  "spectrum/llama-3-2-1B-config.yaml"
    yaml_file = "spectrum/gemmina_7b_it_spect_25.yaml"
    try:
        with open(yaml_file, 'r') as file:
            yaml_content = file.read()

        # Load YAML content into a dictionary
        my_dict = load_yaml_as_dict(yaml_content)

        # Print the resulting dictionary
        # print(yaml_dict)

    except FileNotFoundError as fnf_error:
        print(f"Error: {fnf_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    # print(yaml_dict)
    keys =    my_dict.keys()

    # modellist = ["meta-llama/Meta-Llama-3.1-8B-Instruct"]
    # modellist = ["mistralai/Mistral-7B-Instruct-v0.3"]
    modellist = ["google/gemma-7b-it"]
    #EleutherAI/pythia-160m,dtype="bfloat16",revision=step143000
    # my_dict= torch.load('spectrum/layer_selection_25p_.pt')

    print(len(my_dict))
    print(list(my_dict.keys()))
    # exit()
    # qs=['mlp.down_proj layers', 'mlp.gate_proj layers', 'mlp.up_proj layers', 'self_attn.k_proj layers',
    #  'self_attn.o_proj layers', 'self_attn.q_proj layers', 'self_attn.v_proj layers']
    # kk = "self_attn.v_proj layers"
    #../Datasets/llama_weights/base_llama8b_8_spec_top_25_down_proj_.pt
    #../Datasets/llama_weights/base_llama8b_8_spec_top_25_gate_proj_.pt
    #../Datasets/llama_weights/base_llama8b_8_spec_top_25_up_proj_.pt
    # del my_dict[qs[0]]
    # del my_dict[qs[1]]
    # del my_dict[qs[2]]
    # del my_dict[qs[4]]
    # del my_dict[qs[5]]

    print(list(my_dict.keys()))
    wl = []

    weights = {}
    mw ={}
    ws =[]
    i=0
    # layers = my_dict[qs[2]]
    layers = list(my_dict.values())
    layers =[item for layer in layers for item in layer]

    print(len(layers))
    print(layers[0])
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
        # print(list(std))
        # exit()

        w, we = get_weights_from_list(std, layers)
        # we = torch.cat(we, -1)
        # w, we = extract_layers_weights(std)
        ws.append(we)

        weights.update(w)
        print(len(weights))
#
    # torch.save(weights, f'../Datasets/llama_weights/base_mistral_7b_spec_top_1_ffn_.pt')
    torch.save(weights, f'../Datasets/llmdata/gemini_7b_int_top_25p_attn_.pt')
    #'../Datasets/llama_weights/base_llama8b_top_8_spec_ffn.pt
    #

