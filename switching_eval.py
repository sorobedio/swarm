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



def load_model_config(file):
    with open(file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def my_loss(output, target):
    ed = 0
    n = 256 * 256 * 3
    step = 8192
    loss = 0.0
    for i in range(n, step):
        ed = i + step
        loss += F.mse_loss(output[:, i:ed] - target[: i:ed]) / (torch.std(output[:, i:ed]))
    # loss = torch.mean((output[:, i:ed] - target[: i:ed])**2)
    return loss


# my_loss  = my_loss()

def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def train(model, optimizer, n_epochs, traindataloader, testdataloader=None):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    bloss = 7000
    btest = 2.0
    cr = []
    # schedulers = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, 5)
    # scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for epoch in range(n_epochs):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0
        total = 0
        idx = 0
        for batch_idx, inputs in enumerate(traindataloader):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                loss, logs = model.training_step(inputs, batch_idx)
            optimizer.zero_grad()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total += inputs['weight'].size(0)
            progress_bar(batch_idx, len(traindataloader), 'Loss: %.6f |'
                         % (train_loss / (batch_idx + 1)))
            idx = batch_idx + 1
        tloss = (train_loss / idx)
        # btst = evaluate(model, traindataloader)
        # print(f'current best test avg  loss: {btest}')
        # if btest > btst:
        #     btest = btst
        #     print(f'new best valid avg loss: {btst}')
        #     torch.save(model,  os.path.join(args.save_path,f'best_valid_loss_llama3-8b_uns.pth'))
        if bloss > tloss:
            bloss = tloss
            print(f'best training loss is:{bloss}')
            torch.save(model, os.path.join(args.save_path, f'best_trainingloss_llama3-8b_head.pth'))
        print(f'best training loss is:{bloss}')


def evaluate(model, testdataloader):
    model.eval()
    test_loss = 0
    idx = 1
    with torch.no_grad():
        for batch_idx, inputs in enumerate(testdataloader):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                loss = model.validation_step(inputs, batch_idx)
            # inputs = inputs.to(device)
            outputs, _ = model(inputs)
            recon_error = F.mse_loss(outputs, inputs) * 1000
            loss = recon_error
            test_loss += loss.item()
            progress_bar(batch_idx, len(testdataloader), 'Loss: %.6f |'
                         % (test_loss / (batch_idx + 1)))
            idx = batch_idx + 1
        tloss = (test_loss / idx)
    return tloss


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


from stage1.modules.losses.CustomLosses import LayerWiseReconLoss, ChunkWiseReconLoss



def apply_operation(df, threshold, maximum, task='leaderboard_gpqa'):
    """
    Apply operation (value - threshold) / (maximum - threshold) on the 'value' column
    and store the results in a new column 'norm_value'.
    If threshold > value, store 0.0 in 'norm_value'.
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    threshold (float): The threshold value.
    maximum (float): The maximum value.

    Returns:
    pd.DataFrame: The DataFrame with the new column 'norm_value' and an additional average.
    """
    if task == 'leaderboard_gpqa':
        cls = 'acc_norm,none'
    elif task == 'leaderboard_bbh':
        cls = 'acc_norm,none'
    elif task == 'leaderboard_ifeval':
        cls = 'acc_norm,none'
    elif task == 'leaderboard_gpqa':
        cls = 'acc_norm,none'
    elif task == 'leaderboard_gpqa':
        cls = 'acc_norm,none'

    def operation(val):
        if threshold > val:
            return 0.0
        return (val - threshold) / (maximum - threshold)

    df['norm_value'] = df[cls].apply(operation)

    # Calculate the average of the 'norm_value' column
    norm_value_avg = df['norm_value'].mean()

    return df, norm_value_avg

def  extract_results(result_dict):
    subtasks = list(result_dict)
    return subtasks
# from datetime import datetime
from helpers.helpers import *
import lm_eval
# from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval import evaluator, utils
from lm_eval.evaluator import request_caching_arg_to_dict
from lm_eval.loggers import EvaluationTracker, WandbLogger
from lm_eval.tasks import TaskManager
from lm_eval.utils import handle_non_serializable, make_table, simple_parse_args_string
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.models.huggingface import HFLM
import random
import numpy as np
import pandas as pd
from tqdm import tqdm



# def encode_and_sample(model, )
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"



if __name__ == "__main__":

    default_seed_string = "0,1234,1234,1234"

    # random_seed= default_seed_string[0]
    # numpy_random_seed= default_seed_string[1]
    # torch_random_seed= default_seed_string[2]
    # fewshot_random_seed= default_seed_string[3]
    random_seed = 0
    numpy_random_seed = 1234
    torch_random_seed = 1234
    fewshot_random_seed = 1234

    seed_message = []
    if random_seed is not None:
        # See https://github.com/EleutherAI/lm-evaluation-harness/pull/1412
        seed_message.append(f"Setting random seed to {random_seed}")
        random.seed(int(random_seed))

    if numpy_random_seed is not None:
        seed_message.append(f"Setting numpy seed to {numpy_random_seed}")
        np.random.seed(int(numpy_random_seed))

    if torch_random_seed is not None:
        seed_message.append(f"Setting torch manual seed to {torch_random_seed}")
        torch.manual_seed(int(torch_random_seed))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('=============loading model================')

    # modellist = ["HuggingFaceTB/SmolLM2-135M-Instruct", 'HuggingFaceTB/SmolLM2-135M']

    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"

    weightlist = torch.load( '../Datasets/llmdata/SmolLM2-heads_.pt')
    # weights = weightlist["SmolLM2-135M-Instruct"]
    print(list(weightlist))
    weights = weightlist['SmolLM2-135M-Instruct']

    # exit()




    print("============================================================")
    # layers = list(weights)
    # print(layers)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 # revision='step143000',
                                                 # attn_implementation="flash_attention_2",
                                                 torch_dtype=torch.bfloat16,
                                                 device_map=device,
                                                 )


    for i in range(1):
        wr = {}
        layers= 'lm_head'
        for l in layers:
            # wr[l] = slerp(0.90, weights[l], wd[l][i])
            wr=weights.reshape(-1)
            # wr[l] = w
            # print(w.shape, w.min(), w.max())

        # model = set_mlp_weights(model, wr)
        std = model.state_dict()
        # # for w in ws:
        std =   set_layer_state_dict(std, wr, layer='lm_head')
        # set_layers_state_dict
        model.load_state_dict(std)

        print('---------evaluating model-----------------------------')

        lm_eval_model = HFLM(model, device=device,
                             batch_size=8,
                             tokenizer=tokenizer,
                             # dtype=torch.bfloat16,
                             )

        task_manager = lm_eval.tasks.TaskManager()
        #
        results = lm_eval.simple_evaluate(  # call simple_evaluate
            model=lm_eval_model,
            tasks=["arc_challenge"],
            # num_fewshot=0,
            # apply_chat_template=True,
            # fewshot_as_multiturn=True,
            # output_base_path="results_Out",
            task_manager=task_manager,
        )

        print('==================================')
        mtable =  make_table(results)
        print(mtable)
        # Get the current date and format it
        current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create the header with asterisks and the date
        header = f"\n{'*' * 40}\n{current_date}\n{'*' * 40}\n"
        # file_path = 'logfiles/untrain_tester_results_top1.txt'

        # df = pd.DataFrame.from_dict(results['results'], orient='index')
        # df = df.dropna(subset=['acc_norm,none'])
        # tdf, acc = apply_operation(df, 0.25, 1.0)
        # # print(tdf)
        # print(f'average results is:{acc}')
        # res = f'average results is:{acc}'

        # Append the header and the Markdown table to the file
        # with open(file_path, 'a') as file:
        #     file.write(header)
        #     file.write(mtable)
        #     file.write('\n')  # Add a new line at the end

        print('****************************************')
        if "groups" in results:
            print(make_table(results, "groups"))
        print(results['results'])
        # logging.info('****************************************')
        # logging.info(f'results={results['results']}')
        # acc = results['results']['winogrande']['acc,none']
        # wacc.append(acc)
    # torch.save(wacc, 'vae_embedding_top_kv_chunks_acc.pt')
    #     exit()

# - leaderboard_bbh (3)
# - leaderboard_gpqa(0)
# - leaderboard_ifeval(0)
# - leaderboard_math_hard(4)
# - leaderboard_mmlu_pro(5)
# - leaderboard_musr(0)

# exit()
        # del model
        # del lm_eval_model
        # del results



