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
        default="stage1/configs/layernorm_stage_1_config_kl.yaml",

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
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
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
    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, 5)
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

# from merge_eval import slerp


# def encode_and_sample(model, )
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"



if __name__ == "__main__":
    # log_format = '%(asctime)s %(message)s'
    # save_file="leaderboard_iefl_.txt"
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    #                     format=log_format, datefmt='%m/%d %I:%M:%S %p')
    # fh = logging.FileHandler(save_file)
    # fh.setFormatter(logging.Formatter(log_format))
    # logging.getLogger().addHandler(fh)

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
    # configs = [OmegaConf.load(opt.base)]
    # myconfig = load_config(opt.base)
    # cli = OmegaConf.from_dotlist(unknown)
    # config = OmegaConf.merge(*configs, cli)

    # model = model.to(device)
    # model.device = device

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

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print('=============loading model================')

    # configs = [OmegaConf.load(opt.base)]
    # myconfig = load_config(opt.base)
    # cli = OmegaConf.from_dotlist(unknown)
    # config = OmegaConf.merge(*configs, cli)
    # autoencoder = instantiate_from_config(config.model)

    # autoencoder = torch.load('./autocheckpoints/pythia_160M_.pth', map_location=device)
    #
    # torch.save(autoencoder.state_dict(), f'checkpoints/stage1/pythia_160_143_.ckpt')
    # exit()


    # weights = torch.load('../../Datasets/phi_3_weights/Phi-3-mini-4k-instruct_last_4_fnn_bf16_.pt')
    # model_id = "microsoft/Phi-3-mini-4k-instruct"
    # chunk_size = 393216
    # max_len = 393216
    # model_id= "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # model_id = "meta-llama/Llama-3.2-1B-Instruct"
    # model_id = "EleutherAI/pythia-70m"

    # model_id = "google/gemma-7b-it"
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    weights = torch.load(f'../Datasets/llmdata/llama-3-1-8b_layer_full.pt')
    print(list(weights))
    x_max = 2.9375
    x_min = -0.9140625
    # exit()
    #
    # chunk_size =2362368
    # chunk_size = 1100416
    chunk_size=7842052
    # chunk_size = 2156032
    scale = 1.0
    # chunk_size = 58720256
    # chunk_size = 1048576


    print("============================================================")
    layers = list(weights)
    print(layers)
    # scale = 0.125
    # scale =1.0
    lw ={}
##############################ffn###################################
    # autoencoder = torch.load('./autocheckpoints/Llama-3.2-1B-Inst_top_2tf_.pth', map_location=device)
    # autoencoder = torch.load('./autocheckpoints/llama-3_2-1B_tf-top4_.pth', map_location=device)
    autoencoder = torch.load('./autocheckpoints/llama_full_.pth', map_location='cpu')
    # torch.save(autoencoder.state_dict(), f'checkpoints/stage1/gemmina_llama_norm_.ckpt')
    # torch.save(autoencoder.state_dict(), f'checkpoints/stage1/pythia_160m_ffn_44step.ckpt')


    autoencoder.to(device)
    autoencoder.eval()
    wd ={}

    num_samples = 3
    # latent_shape = (num_samples, 4, 16, 16)
    latent_shape = (num_samples, 4, 32, 32)
    zweights = {}

    for layer in layers:
        # #
        weight = weights[layer]
        # u = torch.mean(weight, dim=1)
        # v = torch.std(weight, dim=1)
        # x_min = weight.min()
        # x_max = weight.max()

        # weight = (weight-u[:, None])/v[:, None]

        # scale=0.0125
        weight = pad_to_chunk_multiple(weight, chunk_size=chunk_size)
        print(weight.shape)
        weight = 2 * (weight - x_min) / (x_max - x_min) - 1
        print(weight.shape)
        # n =weight.shape[-1]

        weight = torch.split(weight, split_size_or_sections=chunk_size, dim=-1)
        # n =len(weight)
        # print(n)

        use_amp = True

        with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=use_amp):
            wl = []
            zp =[]
            for w in tqdm(weight):
                w = w / scale
                w = w.to(device)
                _, x_rec, prior = autoencoder(w)
                # print(prior.mean.shape, prior.std.shape)
                # print(w.shape, x_rec.shape)
                # exit()

                ze = prior.mean + prior.std * torch.randn(latent_shape).to(device)
                zs = ze.detach().cpu().float()
                # zp.append(zs)
                x_rec =  autoencoder.decode(ze)
                #
                wl.append(x_rec.detach().cpu())
        # zweights[layer] = torch.cat(zp, dim=1).reshape(num_samples, -1)
        # print(len(wl))

        ws = torch.cat(wl, dim=-1) * scale
        print(ws.shape)
        # # ws = ws * v[:, None] + u[:, None]
        ws = 0.5*(ws +1)* (x_max-x_min) + x_min
        wd[layer]=ws
        # # wd[layer] = slerp(0.5, weights[layer], ws)
        # # lw[layer]=ws
    print('finished encoding=========================================')

    # exit()
    #     #

        #
    # we = torch.cat(wg, dim=-1)
    # torch.save(zweights, 'vae_')
    # exit()
    del autoencoder


    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 # revision='step143000',
                                                 # attn_implementation="flash_attention_2",
                                                 torch_dtype=torch.bfloat16,
                                                 device_map=device,
                                                 )
    torch.save(wd, 'wdata/sampled_weights_vae_norm.pt')
    # wd = torch.load('./wdata/sampled_1_.pt')

    wacc =[]
    n = ws.shape[0]
    for i in range(n):
        # l='gemma-7b-it'
        wr = {}
        #['gemma-7b-it', 'Llama-3.2-3B-Instruct']
        # for l in layers:
        #     print(f'layer;--{l}---')
        #     # wr[l] = slerp(0.90, weights[l], wd[l][i])
        wr = wd[l][i].reshape(-1)
        # w = ws[i].reshape(-1)

        std = model.state_dict()
        model=set_model_weights(model, w)
        # for w in ws:ws[i
        # std =   set_layer_state_dict(std, wr, layer='norm')
        model.load_state_dict(std)


        # model.load_state_dict(set_layers_state_dict(std, lw))
        # del wd

        # del weight
        print('---------evaluating model-----------------------------')

        lm_eval_model = HFLM(model, device=device,
                             batch_size=4,
                             tokenizer=tokenizer,
                             # dtype=torch.bfloat16,
                             )

        task_manager = lm_eval.tasks.TaskManager()
        #
        results = lm_eval.simple_evaluate(  # call simple_evaluate
            model=lm_eval_model,
            tasks=["winogrande"],
            num_fewshot=5,
            apply_chat_template=True,
            fewshot_as_multiturn=True,
            # output_base_path="results_Out",
            task_manager=task_manager,
        )

        print('==================================')
        mtable =  make_table(results)
        print(mtable)
        # Get the current date and format it
        current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create the header with asterisks and the date
        # header = f"\n{'*' * 40}\n{current_date}\n{'*' * 40}\n"
        # file_path = 'logfiles/winollama3_3b_results_top1.txt'
        #
        # # df = pd.DataFrame.from_dict(results['results'], orient='index')
        # # df = df.dropna(subset=['acc_norm,none'])
        # # tdf, acc = apply_operation(df, 0.25, 1.0)
        # # # print(tdf)
        # # print(f'average results is:{acc}')
        # # res = f'average results is:{acc}'
        #
        # # Append the header and the Markdown table to the file
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
#
# |  Tasks   |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
# |----------|------:|------|-----:|------|---|-----:|---|-----:|
# |winogrande|      1|none  |     5|acc   |↑  |0.5193|±  | 0.014|
#

# |  Tasks   |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
# |----------|------:|------|-----:|------|---|-----:|---|-----:|
# |winogrande|      1|none  |     5|acc   |↑  |0.5067|±  |0.0141|



