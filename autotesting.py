import json
import math
import time
import torch
import datetime
import argparse
# import reward_modeling
from tqdm import tqdm
from datasets import load_dataset
# from tenacity import retry, wait_random_exponential, stop_after_attempt
# from googleapiclient import discovery
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file, save_file
import random
# import vertexai
import warnings

import argparse, os, sys, datetime, glob
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl
from torch.linalg import multi_dot
from packaging import version
# from omegaconf import OmegaConf
import logging
from torch.utils.data import random_split, DataLoader, Dataset
from functools import partial
from PIL import Image
from helpers.helpers import *
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
from zoodatasets.basedatasets import ZooDataset
from zoodatasets.chunkdatasets import ZooDataset
# from zoodatasets.autodatasets import ZooDataset
# from zoodatasets.chunkdatasets import ZooDataset
# from zoodatasets.tinydatasets import ZooDataset
from helpers.misc import progress_bar
# from data.base import Txt2ImgIterableBaseDataset
from utils.util import instantiate_from_config

from torch.optim import lr_scheduler
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def load_model_config(file):
    with open(file, 'r') as file:
        config = yaml.safe_load(file)
    return config


# from vertexai.generative_models import GenerativeModel, GenerationConfig, SafetySetting, HarmCategory, \
#     HarmBlockThreshold

ICL_PROMPT = None
model = None
tokenizer = None


ONLY_ONE_OR_TWO = None

# determisnistic model behavior for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# safety_config = [
#     SafetySetting(
#         category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
#         threshold=HarmBlockThreshold.BLOCK_NONE,
#     ),
#     SafetySetting(
#         category=HarmCategory.HARM_CATEGORY_HARASSMENT,
#         threshold=HarmBlockThreshold.BLOCK_NONE,
#     ),
#     SafetySetting(
#         category=HarmCategory.HARM_CATEGORY_UNSPECIFIED,
#         threshold=HarmBlockThreshold.BLOCK_NONE,
#     ),
#     SafetySetting(
#         category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
#         threshold=HarmBlockThreshold.BLOCK_NONE,
#     ),
#     SafetySetting(
#         category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
#         threshold=HarmBlockThreshold.BLOCK_NONE,
#     ),
# ]


def lora_weight_visualize(path):
    state_dict_this = load_file(path, device="cpu")
    return [state_dict_this["base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight"][0][0].item(),
            state_dict_this["base_model.model.model.layers.10.self_attn.k_proj.lora_B.weight"][0][0].item()]


# # print(lora_weight_visualize("search/human_{asian_food}_{0.3}_{0.2}_{0.5}_{0.05}_{0.6}_a100-16-4-bk-2/particle_3/personal_best/adapter_model.safetensors"))
#
# def parse_gemini_score(response):
#     # It should be "Rating: ?/10"
#     if "Rating: " not in response:
#         warnings.warn("Gemini score parsing error for string: " + response)
#         return 1  # output format invalid
#     score = response.split("Rating: ")[1].split("/10")[0]
#     return int(score)
#
#
# # print(parse_gemini_score("Rating: 2/10"))
#
# @retry(wait=wait_random_exponential(multiplier=1, max=30), stop=stop_after_attempt(10))
# def gemini_eval(prompt, response):
#     eval_prompt = "For the following instruction and response, please rate the response on a scale of 1 to 10, where 1 is the worst and 10 is the best. Please respond with \"Rating: ?/10\" first and then provide your reason.\n\n + \
#     Instruction: " + prompt + "\n\nResponse: " + response
#     eval_response = gemini_model.generate_content(eval_prompt, generation_config=generationConfig,
#                                                   safety_settings=safety_config).text
#
#     time.sleep(1)
#
#     return parse_gemini_score(eval_response)
#
#
# # print(gemini_eval("Please tell me about your experience with the product.", "I just can't stop using this phone case! It has a very nice texture and also very light, perfect for long trips."))

def update_only_one_or_two(value):
    global ONLY_ONE_OR_TWO
    ONLY_ONE_OR_TWO = value


def curret_time_string():
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return current_time
#
#
# # metric for AbstainQA, check out https://arxiv.org/abs/2402.00367
def get_effective_reliability(correct_flags, abstain_flags):
    answered_correct = 0
    answered_incorrect = 0

    for i in range(len(correct_flags)):
        if abstain_flags[i] == 0:
            if correct_flags[i] == 1:
                answered_correct += 1
            else:
                answered_incorrect += 1

    return (answered_correct - answered_incorrect) / len(correct_flags)

#
# @retry(stop=stop_after_attempt(5), wait=wait_random_exponential(multiplier=1, max=20))
# def get_toxicity_score(text):
#     text = text.split("\n")[0].strip()
#     analyze_request = {
#         'comment': {'text': text},
#         'requestedAttributes': {'TOXICITY': {}}
#     }
#
#     response = client.comments().analyze(body=analyze_request).execute()
#     return response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]


def multiple_choice_prompt(instance_dict, dataset):
    prompt = "Question: " + instance_dict["question"] + "\n"

    if dataset == "knowledge_crosswords":
        prompt = prompt
    elif dataset == "hellaswag":
        prompt = "Please choose an option that best completes the sentence.\n" + prompt
    else:
        prompt = "Please choose an option that best answers the question.\n" + prompt

    for key in instance_dict["choices"].keys():
        prompt += (key + ": " + instance_dict["choices"][key] + "\n")

    prompt += "The answer is"

    # the format of Knowledge Crosswords is more complex and neccesitates an in-context example
    if dataset == "knowledge_crosswords":
        prompt = ICL_PROMPT + "\n" + prompt

    # print(prompt)

    return prompt


def multiple_choice_answer_parsing(instance_dict, output_text):
    # print(output_text)
    # print("-----")

    # directly answer
    for key in instance_dict["choices"].keys():
        if key in output_text[:5]:
            return key
    # "The answer is ."
    for key in instance_dict["choices"].keys():
        if key in output_text[-5:]:
            return key
    # answer text exact match
    for key in instance_dict["choices"].keys():
        if instance_dict["choices"][key].lower() in output_text.lower():
            return key
    return "Z"  # so that it is absolutely incorrect


# for objective 3: reward models and objective 4: human interests we employ chat templates for conversation-like generation
def batch_generate_chat_template(model, tokenizer, prompts, gpu_id, batch_size=10, max_new_tokens=512):
    outputs = []
    # batch_size argument is useless here, sequential generation is necessary
    for prompt in tqdm(prompts):
        chat = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        output = model.generate(input_ids=inputs.to(model.device), max_new_tokens=max_new_tokens, do_sample=False)
        outputs.append(tokenizer.decode(output[0][len(inputs[0]):], skip_special_tokens=True).strip())
    # print(outputs[-1])
    return outputs


# for objective 1: single task and objective 2: multi-task domains we directly generate
def batch_generate(model, tokenizer, prompts, gpu_id, batch_size=10, max_new_tokens=10):
    num_batches = math.ceil(len(prompts) / batch_size)
    outputs = []
    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]

        input_ids = tokenizer(batch_prompts, return_tensors="pt", padding=True).input_ids.to(f"cuda:{gpu_id}")
        output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)

        for j in range(len(output)):
            outputs.append(tokenizer.decode(output[j][len(input_ids[j]):], skip_special_tokens=True).strip())

        del input_ids, output
        torch.cuda.empty_cache()

    return outputs


# given a model, evaluate it on the utility function and return the scalar value
def evaluate(model_path, eval_type, dataset, gpu_id, base_model="google/gemma-7b-it", save_dev_flag=False,
             only_one_or_two=None, skip_flag=False):
    if skip_flag:
        return None

    global model
    global tokenizer
    only_one_or_two = ONLY_ONE_OR_TWO
    # try:
    #     model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)
    #     model.load_adapter(model_path)
    #     model.to(f"cuda:{gpu_id}")
    #     tokenizer = AutoTokenizer.from_pretrained(base_model)
    # except:
    #     del model
    #     del tokenizer
    # model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    # model.to(f"cuda:{gpu_id}")
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # tokenizer.pad_token = tokenizer.eos_token



    # task 1: single task, multiple choice questions
    if eval_type == "multiple_choice":
        eval_data = json.load(open("data/eval/" + dataset + ".json"))["dev"]
        golds = []
        preds = []
        global ICL_PROMPT
        try:
            # in case an ICL prompt is provided for datasets such as Knowledge Crosswords
            # you can provide your own ICL prompt in the dataset json file following knowledge_crosswords.json
            ICL_PROMPT = json.load(open("data/eval/" + dataset + ".json"))["icl_prompt"]
        except:
            pass

        prompts = []
        for question in eval_data:
            prompt = multiple_choice_prompt(question, dataset)
            prompts.append(prompt)

        # hard-defined batch_size for multiple choice questions, reduce if OOM
        BATCH_SIZE = 10

        # change max_new_tokens to larger values for intermediate reasoning
        outputs = batch_generate(model, tokenizer, prompts, gpu_id, batch_size=BATCH_SIZE, max_new_tokens=10)

        for question, output in zip(eval_data, outputs):
            # optionally uncomment the following lines to re-generate for multiple-choice questions that didn't get answered within 10 tokens
            # if multiple_choice_answer_parsing(question, output) == "Z":
            #     output = batch_generate(model, tokenizer, prompts, gpu_id, max_new_tokens = 100)
            golds.append(question["answer"])
            preds.append(multiple_choice_answer_parsing(question, output))

        if save_dev_flag:
            with open(model_path + "/golds_dev.json", "w") as f:
                json.dump(golds, f)
            with open(model_path + "/preds_dev.json", "w") as f:
                json.dump(preds, f)

        # print(golds)
        # print(preds)
        # utility function value is the accuracy score of the model on the multiple choice questions
        return accuracy_score(golds, preds)

    # task 1: single task, exact match questions
    elif eval_type == "exact_match":
        eval_data = json.load(open("data/eval/" + dataset + ".json"))["dev"]
        scores = []  # could be 0/1 binary, could be continuous scores

        prompts = []
        for question in eval_data:
            prompts.append(question["question"])

        # hard-defined batch_size for exact match questions, reduce if OOM
        BATCH_SIZE = 10
        if dataset == "nlgraph" or dataset == "nlgraph_mini":
            BATCH_SIZE = 5

        MAX_NEW_TOKENS = 10
        # math reasoning datasets require more tokens for reasoning
        if dataset == "gsm8k":
            MAX_NEW_TOKENS = 200

        outputs = batch_generate(model, tokenizer, prompts, gpu_id, max_new_tokens=MAX_NEW_TOKENS,
                                 batch_size=BATCH_SIZE)

        # retain only the last 5 tokens for number disambiguation
        if dataset == "gsm8k":
            outputs = [" ".join(output.split(" ")[-5:]) for output in outputs]

        # exact match evaluation
        for question, output in zip(eval_data, outputs):
            if question["answer"] in output:
                scores.append(1)
                time.sleep(0.2)
            else:
                scores.append(0)
            # print(question["answer"], output)
            # print("---------------")

        if save_dev_flag:
            with open(model_path + "/scores_dev.json", "w") as f:
                json.dump(scores, f)

        # utility function value is the accuracy score of the model on the exact match questions
        return sum(scores) / len(scores)

    # # task 1: single task, RealToxicityPrompts special case
    # elif eval_type == "external_api":
    #     eval_data = json.load(open("data/eval/" + dataset + ".json"))["dev"]
    #     scores = []  # could be 0/1 binary, could be continuous scores
    #
    #     prompts = []
    #     for instance in eval_data:
    #         prompts.append(instance["prompt"])
    #
    #     BATCH_SIZE = 10
    #     MAX_NEW_TOKENS = 20
    #
    #     outputs = batch_generate(model, tokenizer, prompts, gpu_id, max_new_tokens=MAX_NEW_TOKENS,
    #                              batch_size=BATCH_SIZE)
    #
    #     if dataset == "realtoxicityprompts" or dataset == "realtoxicityprompts_mini":
    #         for output in tqdm(outputs):
    #             if len(output) < 10:
    #                 scores.append(0.5)  # too short
    #                 continue
    #             try:
    #                 # the higher the better, so 1 - toxicity score
    #                 scores.append(1 - get_toxicity_score(output))
    #             except:
    #                 scores.append(0.5)  # error on the API side
    #             time.sleep(0.9)

        # if save_dev_flag:
        #     with open(model_path + "/scores_dev.json", "w") as f:
        #         json.dump(scores, f)
        #
        # # utility function value is the average anti-toxicity score of the model on the RealToxicityPrompts dataset
        # return sum(scores) / len(scores)

    # task 1: single task, AbstainQA special case
    elif eval_type == "AbstainQA":
        eval_data = json.load(open("data/eval/" + dataset + ".json"))["dev"]
        golds = []
        preds = []

        prompts = []
        for question in eval_data:
            prompt = multiple_choice_prompt(question, dataset)
            prompts.append(prompt)

        outputs = batch_generate(model, tokenizer, prompts, gpu_id, max_new_tokens=10)

        correct_flags = []
        abstain_flags = []

        for question, output in zip(eval_data, outputs):
            # if multiple_choice_answer_parsing(question, output) == "Z":
            #     output = batch_generate(model, tokenizer, prompts, gpu_id, max_new_tokens = 100)
            golds.append(question["answer"])
            preds.append(multiple_choice_answer_parsing(question, output))
            if golds[-1] == preds[-1]:
                correct_flags.append(1)
            else:
                correct_flags.append(0)

        # self-reflection after answering to get abstain decisions
        new_prompts = [
            prompt + "\nProposed answer: " + output + "\nIs the proposed answer true or false? Directly answer with true or false."
            for prompt, output in zip(prompts, outputs)]

        outputs = batch_generate(model, tokenizer, new_prompts, gpu_id, max_new_tokens=10)

        for output in outputs:
            # print(output)
            if "false" in output.lower():
                abstain_flags.append(1)
            else:
                abstain_flags.append(0)

        if save_dev_flag:
            with open(model_path + "/golds_dev.json", "w") as f:
                json.dump(correct_flags, f)
            with open(model_path + "/preds_dev.json", "w") as f:
                json.dump([1 - flag for flag in abstain_flags], f)

        # print(golds)
        # print(preds)
        # utility function value is the effective reliability of the model on the AbstainQA dataset
        return get_effective_reliability(correct_flags, abstain_flags)


# evaluation on the test set, similar to the dev set evaluation, but kept seperate in case the test eval might be dratiscally different from dev in generalization settings
def evaluate_test(model_path, eval_type, dataset, gpu_id, base_model="google/gemma-7b-it", only_one_or_two=None,
                  obj4_save_generation=False):
    global model
    global tokenizer

    only_one_or_two = ONLY_ONE_OR_TWO

    # try:
    #     model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    #     model.load_adapter(model_path)
    #     model.to(f"cuda:{gpu_id}")
    #     tokenizer = AutoTokenizer.from_pretrained(base_model)
    # except:
    #     del model
    #     del tokenizer
    #     model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    #     model.to(f"cuda:{gpu_id}")
    #     tokenizer = AutoTokenizer.from_pretrained(model_path)
    # tokenizer.pad_token = tokenizer.eos_token

    # prompt = "What is the capital of France? Answer:"
    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    # output = model.generate(input_ids, max_new_tokens=10)
    # print(tokenizer.decode(output[0], skip_special_tokens=True))

    # objective 4: human interests
    if eval_type == "human":
        eval_data = json.load(open("data/eval/" + dataset + ".json"))["test"]
        scores = []

        BATCH_SIZE = 1

        prompts = []
        for obj in eval_data:
            prompts.append(obj["prompt"])

        outputs = batch_generate_chat_template(model, tokenizer, prompts, gpu_id, batch_size=BATCH_SIZE,
                                               max_new_tokens=512)

        for i in tqdm(range(len(prompts))):
            scores.append(gemini_eval(prompts[i], outputs[i]))
            if scores[-1] == None:
                scores[-1] = 1

        # if obj4_save_generation:
        #     save_name = model_path.split("/")[-1] + "_" + eval_type + "_" + dataset
        #     with open("data/outputs/" + save_name + ".json", "w") as f:
        #         json.dump({"outputs": outputs}, f, indent=4)
        #
        # with open(model_path + "/scores.json", "w") as f:
        #     json.dump(scores, f)
        return sum(scores) / len(scores)

    # objective 3: reward models
    if eval_type in ["rm_default", "rm_concise", "rm_verbose", "rm_reverse"]:
        assert dataset == "rm"
        val_data = json.load(open("data/eval/" + dataset + ".json"))["test"]

        # hard-defined batch_size for reward modeling objective, reduce if OOM
        BATCH_SIZE = 10

        prompts = []
        for obj in val_data:
            prompts.append(obj["prompt"])

        outputs = batch_generate(model, tokenizer, prompts, gpu_id, batch_size=BATCH_SIZE, max_new_tokens=200)

        del model
        del tokenizer
        torch.cuda.empty_cache()

        rm_mode = None
        for mode in ["default", "concise", "verbose", "reverse"]:
            if mode in eval_type:
                rm_mode = mode
                break

        pairs = []
        assert len(prompts) == len(outputs)
        for i in range(len(prompts)):
            pairs.append(
                [
                    {"role": "user", "content": prompts[i]},
                    {"role": "assistant", "content": outputs[i]}
                ]
            )

        scores_list = reward_modeling.get_reward_scores(pairs, gpu_id, rm_mode)
        with open(model_path + "/scores.json", "w") as f:
            json.dump(scores_list, f)
        return sum(scores_list) / len(scores_list)

    # task 2: multi-task domains
    elif eval_type == "multitask":
        per_dataset_scores = []
        eval_datasets = multitask_domain_dataset_dict[dataset]
        for eval_dataset in eval_datasets:
            # default multi-task evaluation sets are all MC
            per_dataset_scores.append(evaluate_test(model_path, "multiple_choice", eval_dataset, gpu_id))
        assert len(per_dataset_scores) == 2
        if sum(per_dataset_scores) == 0:
            per_dataset_scores = [0.01, 0.01]  # dummy scores
        harmonic_mean = 2 * per_dataset_scores[0] * per_dataset_scores[1] / (
                    per_dataset_scores[0] + per_dataset_scores[1])
        if only_one_or_two == "one":
            return per_dataset_scores[0]
        elif only_one_or_two == "two":
            return per_dataset_scores[1]
        return harmonic_mean

    # task 1: single task, multiple choice questions
    elif eval_type == "multiple_choice":
        eval_data = json.load(open("data/eval/" + dataset + ".json"))["test"]
        golds = []
        preds = []
        global ICL_PROMPT
        try:
            ICL_PROMPT = json.load(open("data/eval/" + dataset + ".json"))["icl_prompt"]
        except:
            pass

        # hard-defined batch_size for multiple choice questions, reduce if OOM
        BATCH_SIZE = 10
        # if dataset == "medqa" or dataset == "hearsay" or dataset == "citation_prediction_classification":
        #     BATCH_SIZE = 2

        prompts = []
        for question in eval_data:
            prompt = multiple_choice_prompt(question, dataset)
            prompts.append(prompt)

        outputs = batch_generate(model, tokenizer, prompts, gpu_id, batch_size=BATCH_SIZE, max_new_tokens=10)

        for question, output in zip(eval_data, outputs):
            # if multiple_choice_answer_parsing(question, output) == "Z":
            #     output = batch_generate(model, tokenizer, prompts, gpu_id, max_new_tokens = 100)
            golds.append(question["answer"])
            preds.append(multiple_choice_answer_parsing(question, output))
            # print(output)
            # print("----------")

        # print(golds)
        # print(preds)

        # # save golds and preds for later ensemble
        # with open(model_path + "/golds.json", "w") as f:
        #     json.dump(golds, f)
        # with open(model_path + "/preds.json", "w") as f:
        #     json.dump(preds, f)

        return accuracy_score(golds, preds)

    # task 1: single task, exact match questions
    elif eval_type == "exact_match":
        eval_data = json.load(open("data/eval/" + dataset + ".json"))["test"]
        scores = []  # could be 0/1 binary, could be continuous scores

        prompts = []
        for question in eval_data:
            prompts.append(question["question"])

        # hard-defined batch_size for exact match questions, reduce if OOM
        BATCH_SIZE = 10
        if dataset == "nlgraph" or "nlgraph_mini":
            BATCH_SIZE = 5

        MAX_NEW_TOKENS = 10
        # math reasoning datasets require more tokens for reasoning
        if dataset == "gsm8k":
            MAX_NEW_TOKENS = 200

        outputs = batch_generate(model, tokenizer, prompts, gpu_id, max_new_tokens=MAX_NEW_TOKENS,
                                 batch_size=BATCH_SIZE)

        if dataset == "gsm8k":
            outputs = [" ".join(output.split(" ")[-10:]) for output in outputs]

        for question, output in zip(eval_data, outputs):
            if question["answer"] in output:
                scores.append(1)
            else:
                scores.append(0)

        with open(model_path + "/scores.json", "w") as f:
            json.dump(scores, f)

        return sum(scores) / len(scores)

    # task 1: single task, RealToxicityPrompts special case
    elif eval_type == "external_api":
        eval_data = json.load(open("data/eval/" + dataset + ".json"))["test"]
        scores = []  # could be 0/1 binary, could be continuous scores

        prompts = []
        for instance in eval_data:
            prompts.append(instance["prompt"])

        # hard-defined batch_size for RealToxicityPrompts, reduce if OOM
        BATCH_SIZE = 10

        MAX_NEW_TOKENS = 20

        outputs = batch_generate(model, tokenizer, prompts, gpu_id, max_new_tokens=MAX_NEW_TOKENS,
                                 batch_size=BATCH_SIZE)

        if dataset == "realtoxicityprompts" or dataset == "realtoxicityprompts_mini":
            for output in tqdm(outputs):
                if len(output) < 10:
                    scores.append(0.5)  # too short
                    continue
                try:
                    scores.append(1 - get_toxicity_score(output))
                    # if get_toxicity_score(output) >= 0.5: # maybe, maybe no cutoff since its calibrated?
                    #     scores.append(0) # toxic, to maintain the higher the better scores
                    # else:
                    #     scores.append(1)
                except:
                    scores.append(0)  # error on the API side
                time.sleep(0.9)

        with open(model_path + "/scores.json", "w") as f:
            json.dump(scores, f)

        print(model_path, "finished!")

        return sum(scores) / len(scores)

    # task 1: single task, AbstainQA special case
    elif eval_type == "AbstainQA":
        eval_data = json.load(open("data/eval/" + dataset + ".json"))["test"]
        golds = []
        preds = []
        # global ICL_PROMPT
        # try:
        #     ICL_PROMPT = json.load(open("data/eval/" + dataset + ".json"))["icl_prompt"]
        # except:
        #     pass

        prompts = []
        for question in eval_data:
            prompt = multiple_choice_prompt(question, dataset)
            prompts.append(prompt)

        outputs = batch_generate(model, tokenizer, prompts, gpu_id, max_new_tokens=10)

        correct_flags = []
        abstain_flags = []

        for question, output in zip(eval_data, outputs):
            # if multiple_choice_answer_parsing(question, output) == "Z":
            #     output = batch_generate(model, tokenizer, prompts, gpu_id, max_new_tokens = 100)
            golds.append(question["answer"])
            preds.append(multiple_choice_answer_parsing(question, output))
            if golds[-1] == preds[-1]:
                correct_flags.append(1)
            else:
                correct_flags.append(0)

        new_prompts = [
            prompt + "\nProposed answer: " + output + "\nIs the proposed answer true or false? Directly answer with true or false."
            for prompt, output in zip(prompts, outputs)]

        outputs = batch_generate(model, tokenizer, new_prompts, gpu_id, max_new_tokens=10)

        for output in outputs:
            # print(output)
            if "false" in output.lower():
                abstain_flags.append(1)
            else:
                abstain_flags.append(0)

        # with open(model_path + "/golds.json", "w") as f:
        #     json.dump(correct_flags, f)
        # with open(model_path + "/preds.json", "w") as f:
        #     json.dump([1 - flag for flag in abstain_flags], f)

        # print(golds)
        # print(preds)
        return get_effective_reliability(correct_flags, abstain_flags)


def get_parser():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="google/gemma-7b-it",
                        help="Path or name of the model to use")
    parser.add_argument("--eval_type", type=str, default="multiple_choice",
                        help="Type of evaluation to perform")
    parser.add_argument("--dataset", type=str, default="mmlu",
                        help="Dataset to use for evaluation")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="ID of GPU to use")
    parser.add_argument("--base_model", type=str, default="google/gemma-7b-it",
                        help="Base model to use")
    parser.add_argument("--save_dev_flag", action="store_true", default=False,
                        help="Whether to save development data")
    parser.add_argument("--only_one_or_two", type=int, default=None, choices=[1, 2],
                        help="Restrict to processing only one or two items")
    parser.add_argument("--skip_flag", action="store_true", default=False,
                        help="Whether to skip certain processing steps")

    return parser




def add_to_config(mydict, cfl="./Experiments/stage1/configs/base_config_imnet_kl.yaml"):
    with open(cfl, 'w') as configfile:
        data = yaml.dump(mydict, configfile, indent=4, sort_keys=False)
        print("Write successful")


def load_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def m_collate(batch):
    sample = {}

    data = [item['weight'] for item in batch]

    data = torch.cat(data, 0).type(torch.float32)

    return data


def pad_to_chunk_multiple(x, chunk_size):
    shape = x.shape
    if len(shape) < 2:
        x = x.unsqueeze(0)
        shape = x.shape
    max_in = chunk_size * math.ceil(shape[1] / chunk_size)
    if max_in > shape[1]:
        delta1 = max_in - shape[1]
        x = F.pad(x, (0, delta1, 0, 0), "constant", 0)
    return x
from helpers.helpers import *

from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

if __name__=='__main__':

    default_seed_string = "0,1234,1234,1234"



    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('=============loading model================')



    #
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(model_id,
    #                                              # revision='step143000',
    #                                              # attn_implementation="flash_attention_2",
    #                                              torch_dtype=torch.bfloat16,
    #                                              device_map=device,
    #                                              )


    model_names = ["code_alpaca", "cot", "flan_v2", "gemini_alpaca", "lima", "oasst1", "open_orca", "science", "sharegpt", "wizardlm"]
    vect_weights={}

    parser = get_parser()
    args = parser.parse_args()
    model_path = args.model_path
    eval_type = args.eval_type
    dataset = args.dataset
    gpu_id = args.gpu_id
    base_model = args.base_model
    save_dev_flag = args.save_dev_flag
    only_one_or_two = args.only_one_or_two



    # model_names = ["code_alpaca", "cot", "flan_v2", "gemini_alpaca", "lima", "oasst1", "open_orca", "science",
    #                "sharegpt", "wizardlm"]
    weights = torch.load("../Datasets/gemina7b_it_lora_weights.pt")

    # autoencoder = torch.load('./autocheckpoints/lora_hunk_base.pth', map_location='cpu')
    autoencoder = torch.load('./autocheckpoints/lora_hunk_base_full.pth', map_location='cpu')
    # torch.save(autoencoder,'./autocheckpoints/lora_hunk_base_full_copy_v.pth')

    #
    # torch.save(autoencoder.state_dict(), f'checkpoints/stage1/gemmina_lora_.ckpt')
    # exit()

    autoencoder.to(device)
    autoencoder.eval()
    wd = {}
    layers = list(weights)
    layers = layers
    chunk_size = 1376256
    scale =0.1

    num_samples = 1
    # latent_shape = (num_samples, 4, 16, 16)
    latent_shape = (num_samples, 4, 32, 32)
    zweights = {}

    # for layer in layers:
    #     # #
    #     weight = weights[layer]
    #     print(f'---original--shape{weight.shape}')
    #
    #     weight = pad_to_chunk_multiple(weight, chunk_size=chunk_size)
    #     print(weight.shape)
    #     # weight = 2 * (weight - x_min) / (x_max - x_min) - 1
    #     # print(weight.shape)
    #     # n =weight.shape[-1]
    #
    #     weight = torch.split(weight, split_size_or_sections=chunk_size, dim=-1)
    #
    #     use_amp = True
    #
    #     with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=use_amp):
    #         wl = []
    #         zp = []
    #         for w in tqdm(weight):
    #             w = w / scale
    #             w = w.to(device)
    #             _, x_rec, prior = autoencoder(w)
    #             # print(prior.mean.shape, prior.std.shape)
    #             # print(w.shape, x_rec.shape)
    #             # exit()
    #             #
    #             # ze = prior.mean + prior.std * torch.randn(latent_shape).to(device)
    #             # zs = ze.detach().cpu().float()
    #             # # zp.append(zs)
    #             # x_rec = autoencoder.decode(ze)
    #             #
    #             wl.append(x_rec.detach().cpu())
    #     # zweights[layer] = torch.cat(zp, dim=1).reshape(num_samples, -1)
    #     # print(len(wl))
    #
    #     ws = torch.cat(wl, dim=-1) * scale
    #     print(ws.shape)
    #     wd[layer] = ws
    # print('finished encoding=========================================')
    #
    # del autoencoder

    # model_names=list(wd.keys())
    base_model = "google/gemma-7b-it"
    # modelist=list(wd)
    modelist = list(weights)
    print('=====================================================================')
    # print(modelist)
    print('=====================================================================')

    wacc = []
    # weights =wd['gemma-7b-it']
    # weights = wd
    # n= weights.shape[0]
    # print(n)
    utilities =[]
    accs =[]
    results_dict ={}
    for k in modelist:
        ks = "bunsenfeng/" + k
        if k not in model_names:
            model_path = "bunsenfeng/code_alpaca"
        # model = AutoModelForCausalLM.from_pretrained("bunsenfeng/" + model_name)
        else:
            model_path=ks
            # continue
        print(f'-=={k}==--loading--{ks}--model=======')

        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)
        model.load_adapter(model_path)
        print('---base--model--lora')
        model.to(f"cuda:{gpu_id}")
        tokenizer = AutoTokenizer.from_pretrained(base_model)



        tokenizer.pad_token = tokenizer.eos_token

        # wr = wd[k].reshape(-1)
        wr = weights[k].reshape(-1)
        std = model.state_dict()

        # for w in ws:ws[i
        std = set_layer_state_dict(std, wr, layer='lora')
        model.load_state_dict(std)



        # model.load_state_dict(set_layers_state_dict(std, lw))
        # del wd

        # results = evaluate(model_path, eval_type, dataset, gpu_id, base_model="google/gemma-7b-it", save_dev_flag=False,
        #          only_one_or_two=None, skip_flag=False)

        # results =results*100.0
        # # utilities.append(results)
        # print(results)
        # # print('-----evaluated======================================')
        acc =evaluate_test(model, eval_type, dataset, gpu_id, base_model="google/gemma-7b-it", only_one_or_two=None,
                      obj4_save_generation=False)
        results_dict[k] = acc*100.0
        # print(acc*100.0)
        print(f'=========={k}============{acc*100}==============================')
    print(results_dict)
        # accs.append(acc*100)
    # torch.save(utilities, 'wdata/utilities_mdt_norm_gsm8k.pt')
    # print(sorted(accs, reverse=True))

