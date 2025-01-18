import os
import shutil
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft.utils.save_and_load import set_peft_model_state_dict, get_peft_model_state_dict


def lora_merge(weights, lora_name_list, output_name):

    # the fast merge: load only state_dicts, merge them, save only state_dicts
    # apply to the setting that models share the same architecture, sharding, and adapter format
    lora_state_dict_list = []
    for lora_name in lora_name_list:
        state_dict_this = load_file(os.path.join(lora_name, "adapter_model.safetensors"), device="cpu")
        lora_state_dict_list.append(state_dict_this)

    final_state_dict = {}
    for i in range(len(lora_state_dict_list)):
        if i == 0:
            for key in lora_state_dict_list[i].keys():
                final_state_dict[key] = weights[i] * lora_state_dict_list[i][key]
        else:
            for key in lora_state_dict_list[i].keys():
                assert key in final_state_dict.keys()
                final_state_dict[key] += weights[i] * lora_state_dict_list[i][key]

    if not os.path.exists(output_name):
        os.mkdir(output_name)
    save_file(final_state_dict, os.path.join(output_name, "adapter_model.safetensors"))

    return final_state_dict