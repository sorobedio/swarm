# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import argparse, os, sys, datetime, glob
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl

from packaging import version
# from omegaconf import OmegaConf
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from dit_diffusion import create_diffusion
from utils.util import instantiate_from_config
# from diffusers.models import AutoencoderKL
# from download import find_model
from models import DiT_models
import argparse
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


# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(description='Autoencoder Training')
    parser.add_argument('--data', default='../../Datasets/llama3_weights', type=str, help='dataset root')
    parser.add_argument('--data_root', default='../Datasets/', type=str, help='dataset root for cifar10, mnist, ..')
    parser.add_argument('--topk', default=30, type=int, help='number of sample per dataset in training loader')
    parser.add_argument('--dataset', default='joint', type=str, help='dataset choice amoung'
                                                                     ' [mnist, svhn, cifar10, stl10, joint')
    parser.add_argument('--split', default='train', type=str, help='dataset split{ train, test, val]')
    parser.add_argument('--ae_type', default='ldm', type=str, help='auto encoder type [ldm, vqvae, simple]')
    parser.add_argument('--save_path', default='autocheckpoints', type=str, help='checkpointys folders')
    parser.add_argument('--gpus', default=0, type=int, help='device')
    # parser.add_argument('--num_workers', default=4, type=int, help='device')
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default='DiT-S/2')
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--print-freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--half', dest='half', action='store_true',
                        help='use half-precision(16-bit) ')
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")

    parser.add_argument('--n_epochs', default=1000000, type=int, help='max epoch')
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="adt",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        # default="stage1/configs/first_stage_base_config_kl.yaml",
        # default="stage2/configs/norm_base_config.yaml",
        # default="stage2/configs/small_norm_base_config.yaml",
        default="dit/configs/dit_layer_base_config.yaml",  # vit_lora_config_kl.yaml

        # default="stage1/configs/loara_base_config_kl.yaml",

    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    # parser.add_argument(
    #     "-s",
    #     "--seed",
    #     type=int,
    #     default=23,
    #     help="seed for seed_everything",
    # )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    return parser


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


import torch.nn as nn

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
# from omegaconf import OmegaConf
from merge_eval import slerp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default='DiT-S/2')
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    # parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--print-freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--half', dest='half', action='store_true',
                        help='use half-precision(16-bit) ')
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument('--data', default='../Datasets/hyperzoo', type=str, help='dataset root')
    parser.add_argument('--split', default='train', type=str, help='dataset split{ train, test, val]')
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        # default="stage2/configs/small_norm_base_config.yaml",
        default="dit/configs/dit_layer_base_config.yaml",  # vit_lora_config_kl.yaml
        # default="stage2/configs/norm_base_config.yaml",
        # default="stage2/configs/small_norm_base_config.yaml",
    )



    criterions = nn.CrossEntropyLoss().cuda()

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # if args.ckpt is None:
    #     assert args.model == 'DiT-S/2', "Only DiT-XL/2 models are available for auto-download."
    #     assert args.image_size in [256, 512]
    #     assert args.num_classes == 2



    opts, unknown = parser.parse_known_args()
    # parser = Trainer.add_argparse_args(parser)

    # nowname = opt.name + now
    # seed_everything(opt.seed)
    print(opts.base)
    print('----------------------')
    # configs = [OmegaConf.load(opts.base)]
    # # myconfig = load_config(opt.base)
    # cli = OmegaConf.from_dotlist(unknown)
    # config = OmegaConf.merge(*configs, cli)
    # vae = instantiate_from_config(config.model).to(device)
    # vae.eval()
    # for p in vae.parameters():
    #     p.requires_grad = False

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # sys.path.append(os.getcwd())
    parser = get_parser()
    use_amp = True

    args = parser.parse_args()
    opt, unknown = parser.parse_known_args()

    # device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

    nowname = opt.name + now
    # seed_everything(opt.seed)
    print(opt.base)
    print('----------------------')

    default_seed_string = "0,1234,1234,1234"

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


    print('=============loading model================')

    config = load_config(opt.base)
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    conds = torch.load('../../Datasets/llama_weights/base_mini_llama_normlabels_.pt')
    # conds = torch.load('../../Datasets/llama3_weights/chunk_attn_llma3_labels.pt')
    # chunk_size =1048576
    chunk_size = 4096
    # chunk_size=524288
    scale = 0.1

    print("============================================================")
    layers = list(conds)

    lw = {}

    # ###################norm mini layers######################
    ldmmodel = instantiate_from_config(config['model'])
    # ldmmodel = instantiate_from_config(config.model)
    # std = torch.load('./ldm_checkpoints/checkpoint_model_layer_norm_llama3-1-8b_epoch=5904_.ckpt')['state_dict']

    std = torch.load('./dit_checkpoints/checkpoint_dit_llama_3_2_1b_tp_1_blk_epoch=61993_.ckpt')['state_dict']
    ldmmodel.load_state_dict(std)
    ldmmodel = ldmmodel.to(device)
    ldmmodel.to(device)
    ldmmodel.eval()
    wd = {}
    num_samples = 15
    batches = 15
    n =  num_samples // batches
    weight_dicts={}

        # latent_shape = (num_samples, 4, 8, 8)
    weights=None
    for layer in layers:
        # weights =
        # for i in range(n):
        # latent_shape = (batche, 4, 16, 16)
        xc = [conds[layer]] * num_samples
        xc = torch.tensor(xc, device=device)
        samples = ldmmodel.condsample(y=xc)
        weights = samples.detach().cpu() * scale
        # weights.append(ws)
    # weights = torch.cat(weights, dim=0)
    wd[layer] = weights.type(torch.bfloat16)
    print(f'finished encoding=========================================')
    torch.save(wd, 'wdata/dit_sampled_weights_1000_norm.pt')
    del ldmmodel
    del xc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map=device,
                                                 )

    # wd = torch.load('./wdata/sampled_1_.pt')

    for i in range(num_samples):
        wr = {}
        for l in layers:
            # wr[l] = slerp(0.0, weights[l], wd[l][i])
            wr[l] = wd[l][i].reshape(-1)

        std = model.state_dict()
        # for w in ws:
        std = set_layers_state_dict(std, wr)
        model.load_state_dict(std)
        # model.load_state_dict(set_layers_state_dict(std, lw))
        # del wd
        # --apply_chat_template \
        # - -fewshot_as_multiturn \
        # attn_implementation="flash_attention_2"\
        # --tasks leaderboard_musr\
        # del weight
        print('---------evaluating model-----------------------------')

        lm_eval_model = HFLM(model, device=device, batch_size='auto')

        task_manager = lm_eval.tasks.TaskManager()
        #
        results = lm_eval.simple_evaluate(  # call simple_evaluate
            model=lm_eval_model,
            tasks=["leaderboard_gpqa"],
            num_fewshot=0,
            apply_chat_template=True,
            # fewshot_as_multiturn=True,
            # output_base_path="results_Out",
            task_manager=task_manager,
        )

        print('==================================')
        mtable = make_table(results)
        print(mtable)
        # Get the current date and format it
        current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create the header with asterisks and the date
        header = f"\n{'*' * 40}\n{current_date}\n{'*' * 40}\n"
        file_path = 'logfiles/ditsimplellama3_3.1_results_norm_musr.txt'
        # Append the header and the Markdown table to the file
        with open(file_path, 'a') as file:
            file.write(header)
            file.write(mtable)
            file.write('\n')  # Add a new line at the end

        print('****************************************')
        if "groups" in results:
            print(make_table(results, "groups"))
        print(results['results'])
        # logging.info('****************************************')
        # logging.info(f'results={results['results']}')
        # exit()
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


