import argparse
import os
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import yaml


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



def get_basic_blk_weights(model, nbk=7, model_name=None):
    weights = {}

    # Loop through the number of transformer blocks (nbk)
    for i in range(nbk):
        # Define the layer path
        layer_path = f'model.encoder.layers.encoder_layer_{i}'

        # If model_name is provided, append it to the key
        if model_name is not None:
            key_prefix = f'{model_name}-model.encoder.layers.encoder_layer_{i}'
        else:
            key_prefix = f'model.encoder.layers.encoder_layer_{i}'

        # Access the transformer block by evaluating its path dynamically
        block_layer = eval(layer_path)

        # Get the state_dict of the transformer block
        std = block_layer.state_dict()

        # Assuming `gets_weights` is a function that processes the state_dict
        w = gets_weights(std)
        print(f'-------------{w.shape}-------------')

        # Store the weights in the dictionary with the prefixed key
        weights[key_prefix] = w

    return weights


def add_to_config(mydict, cfl):
    with open(cfl, 'a') as configfile:
        data = yaml.dump(mydict, configfile, indent=4)
        print("Write successful")

def load_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def set_state_dict(std, weights):
    # std = model.state_dict()
    st = 0
    for params in std:
        if not params.endswith('num_batches_tracked'):
            shape = std[params].shape
            device = std[params].device
            dtp = std[params].dtype
            ed = st + np.prod(shape)
            std[params] = weights[st:ed].reshape(shape).type(dtp).to(device)
            # model.load_state_dict(std)
            st = ed
    return std


def set_norm_state_dict(std, weights, tg='norm'):
    # std = model.state_dict()
    st = 0
    for params in std:
        if not params.endswith('num_batches_tracked'):
            if tg in params:
                shape = std[params].shape
                device = std[params].device
                dtp = std[params].dtype
                ed = st + np.prod(shape)
                std[params] = weights[st:ed].reshape(shape).type(dtp).to(device)
                # model.load_state_dict(std)
                st = ed
    return std


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

def set_layer_state_dict(std, weights, layer='mlp'):
    # std = model.state_dict()
    st = 0
    for params in std:
        if not params.endswith('num_batches_tracked'):
            if layer in params:
                shape = std[params].shape
                device = std[params].device
                dtp = std[params].dtype
                ed = st + int(np.prod(shape))
                # print(ed)

                std[params] = weights[st:ed].reshape(shape).type(dtp).to(device)
                # model.load_state_dict(std)
                st = ed
                # print(f'setting parameters---{params}')
    return std


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


def set_layers_state_dict(std, weights):
    # std = model.state_dict()
    layers = list(weights)
    # st = 0
    for params in layers:
        if not params.endswith('num_batches_tracked'):
            shape = std[params].shape
            device = std[params].device
            dtp = std[params].dtype
            st =0
            ed = st + np.prod(shape)
            std[params] = weights[params][st:ed].reshape(shape).type(dtp).to(device)
            # model.load_state_dict(std)
            # st = ed
    return std


def set_layers_state_dict_ecp(std, weights, cond='norm', tgt='mlp'):
    # std = model.state_dict()
    layers = list(weights)
    # st = 0
    for params in layers:
        if not params.endswith('num_batches_tracked'):
            if cond in params:
                continue
            if tgt in params:
                shape = std[params].shape
                device = std[params].device
                dtp = std[params].dtype
                st =0
                ed = st + np.prod(shape)
                std[params] = weights[params][st:ed].reshape(shape).type(dtp).to(device)
                # model.load_state_dict(std)
                st = ed
    return std



def gets_weights(std):
    # std = model.state_dict()
    weights = []
    for params in std:
        if not params.endswith('num_batches_tracked'):
            if 'mean' in params or 'var' in params:
                continue
            # print(params)
            w = std[params].reshape(-1)
            weights.append(w)
    return torch.cat(weights, -1)


def set_model_weights(model, weights):
    std = model.state_dict()
    st = 0
    for params in std:
        if not params.endswith('num_batches_tracked'):
            if params.endswith('running_var') or params.endswith('running_mean'):
                continue
            elif 'rotary_emb' in params:
                print(f'found------{params}-------------')
                continue
            # elif 'linear' in params:
            #     continue
            shape = std[params].shape
            dtp = std[params].dtype
            device = std[params].device
            ed = st + np.prod(shape)
            std[params] = weights[st:ed].reshape(shape).type(dtp).to(device)
            model.load_state_dict(std)
            st = ed
    return model

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
    ws = torch.cat(ws, dim=-1)
    return weights, ws

def set_weights(model, weights):
    std = model.state_dict()
    st = 0
    for params in std:
        if not params.endswith('num_batches_tracked'):
            shape = std[params].shape
            ed = st + np.prod(shape)
            std[params] = weights[st:ed].reshape(shape)
            model.load_state_dict(std)
            st = ed



def vecpadder(x, max_in=3728761 * 3):
    shape = x.shape
    delta1 = max_in - shape[0]
    x = F.pad(x, (0, delta1))
    return x


def pad_to_chunk_multiple(x, chunk_size):
    shape = x.shape
    if len(shape)<2:
        x =x.unsqueeze(0)
        shape = x.shape
    max_in = chunk_size*math.ceil(shape[1]/chunk_size)
    delta1 = max_in - shape[1]
    # x = F.pad(x, (0, delta1))
    x =F.pad(x, (0, delta1, 0, 0), "constant", 0)
    return x

def matpadder(x, max_in=512):
    shape =x.shape
    # delta1 = max_in - shape[0]
    delta2 = max_in - shape[1]

    out = F.pad(x, (0, delta2, 0, 0), "constant", 0)
    return out