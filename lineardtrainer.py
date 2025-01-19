import argparse, os, sys, datetime, glob
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl
from torch.linalg import multi_dot
from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset
from functools import partial
from PIL import Image
from helpers.helpers import *
# from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.rank_zero import rank_zero_only
# from pytorch_lightning.utilities import rank_zero_info
# from zoodatasets.weightsdatasets import ZooDataset
from zoodatasets.chunkdatasets import ZooDataset
# from zoodatasets.FFNdatasets import ZooDataset
# from zoodatasets.layerdatasets import ZooDataset
# from zoodatasets.basedatasets import ZooDataset
# from zoodatasets.weightsdatasets import ZooDataset
from helpers.misc import progress_bar
# from data.base import Txt2ImgIterableBaseDataset
from utils.util import instantiate_from_config

# train.py
import wandb
import random  # for demo script
from torch.utils.tensorboard import SummaryWriter

# wandb.login()



os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
    parser.add_argument('--data', default='../Datasets', type=str, help='dataset root')
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

        # default="stage1/configs/pythia_70_config_kl.yaml",
        default="stage1/configs/llama_linear_config_kl.yaml",

        #
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
#
def seed_everything(seed=1234):
    import random, os
    import numpy as np
    import torch

    random.seed(0)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
def my_loss(output, target):
    ed =0
    n = 256*256*3
    step = 8192
    loss = 0.0
    for i in range(n, step):
        ed = i+step
        loss+=F.mse_loss(output[:, i:ed] - target[: i:ed])/(torch.std(output[:, i:ed]))
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
    bloss = 100000.0
    btest = 2.0
    cr =[]
    # schedulers = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, 5)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for epoch in range(n_epochs):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0
        total = 0
        idx = 0
        for batch_idx, inputs in enumerate(traindataloader):
            # input()
            optimizer.zero_grad()
            # with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            #     loss, logs = model.training_step(inputs, batch_idx)
            loss, logs = model.training_step(inputs, batch_idx)

            # scaler.scale(loss).backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip
            # scaler.step(optimizer)
            # scaler.update()
            loss.backward()
            optimizer.step()
            # schedulers.step()
            train_loss += loss.item()

            curr_lr = optimizer.param_groups[-1]['lr']
            total += inputs['weight'].size(0)
            progress_bar(batch_idx, len(traindataloader), 'Loss: %.6f |'
                         % (train_loss / (batch_idx + 1)))
            idx = batch_idx + 1
            # schedulers.step()

        tloss = (train_loss / idx)
        # scheduler.step()
        # Log loss and accuracy to TensorBoard
        writer.add_scalar("Loss/train", tloss, epoch)
        # scheduler.step()
        # btst = evaluate(model, traindataloader)
        # print(f'current best test avg  loss: {btest}')
        # if btest > btst:
        #     btest = btst
        #     print(f'new best valid avg loss: {btst}')
        #     torch.save(model,  os.path.join(args.save_path,f'best_valid_loss_llama3-8b_uns.pth'))
        if bloss > tloss:
            bloss = tloss
            print(f'saving best training loss is:{bloss}')
            torch.save(model, os.path.join(args.save_path,f'blcok1_llama_.pth'))
            # torch.save(model.state_dict(), os.path.join(args.save_path, f'llama_3_1_8B_models_ffn_l-30.ckpt'))
        print(f'best training loss is:{bloss}  lr={curr_lr}')

        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         writer.add_histogram(f'track_linear_gradients/{name}', param.grad, global_step=epoch)

# Llama-3.2-1B-Inst_top_2tf_.pth to test gpt2_full_.pth


def evaluate(model , testdataloader):
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
            idx = batch_idx+1
        tloss =(test_loss / idx)
    return  tloss


def add_to_config(mydict, cfl="./Experiments/stage1/configs/base_config_imnet_kl.yaml"):
    with open(cfl, 'w') as configfile:
        data = yaml.dump(mydict, configfile, indent=10, sort_keys=False)
        print("Write successful")

def load_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def m_collate(batch):
    sample = {}

    data = [item['weight'] for item in batch]

    data = torch.cat(data, 0).type(torch.float32)

    return data
def lr_lambda(current_step: int, warmup_iters=50):
    if current_step < warmup_iters:
        return current_step / max(1, warmup_iters)
    return 1.0

from stage1.modules.losses.CustomLosses import LayerWiseReconLoss, ChunkWiseReconLoss
from schedulers.lr_utils import CustomCosineWarmRestartScheduler, WarmUpAndDecayLR

if __name__ == "__main__":

    writer = SummaryWriter(log_dir="linear_llama_testloss/tensorboard_encod")
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # sys.path.append(os.getcwd())
    parser = get_parser()
    use_amp = True

    args = parser.parse_args()
    opt, unknown = parser.parse_known_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainset = ZooDataset(root=args.data,  dataset="joint", split=args.split,
                          scale=1.0, normalize=None)
    # valset = ZooDataset(root=args.data, dataset=args.dataset, split=args.split, normalize=False)
#0.5
    traindataloader = DataLoader(trainset, shuffle=True, batch_size=256, num_workers=4,
                                 # collate_fn=m_collate,
                                 )
    # testdataloader = DataLoader(valset, shuffle=False, batch_size=4, num_workers=4)

    # parser = Trainer.add_argparse_args(parser)

    nowname= opt.name+now
    # seed_everything(opt.seed)
    print(opt.base)
    print('----------------------')
    configs = [OmegaConf.load(opt.base)]
    myconfig= load_config(opt.base)
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    model = instantiate_from_config(config.model)
    model = model.to(device)
    model.device = device
    optimizer = model.configure_optimizers()
    #
    # initial_lr = model.learning_rate
    # # Number of warmup iterations
    # warmup_iters = 50
    # # Number of total iterations (epochs * iterations per epoch)
    # total_iters = 100000
    # # Linear warmup schedulers
    # scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # # Cosine annealing schedulers after warmup
    # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(total_iters - warmup_iters))
    # # Combine schedulers using SequentialLR
    # schedulers = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine],
    #                          milestones=[warmup_iters])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-8, last_epoch=-1)
    # scheduler = CustomCosineWarmRestartScheduler(optimizer, max_lr=, min_lr=1e-8, first_cycle_steps=400,
    #                                  cycle_mult=1, gamma=1.0, warmup_steps=0,
    #                                  last_epoch=-1)
    # scheduler = WarmUpAndDecayLR(optimizer, warmup_steps=200, cosine_steps=200, gamma=0.1, T_mult=1)
    criterion = model.loss
    # train(model, optimizer, args.n_epochs, traindataloader, testdataloader)
    train(model, optimizer, args.n_epochs, traindataloader)

