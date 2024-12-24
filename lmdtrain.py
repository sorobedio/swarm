import argparse, os, sys, datetime, glob
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.util import instantiate_from_config

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
    parser.add_argument('--data_dir', default='../../Datasets/', type=str, help='dataset root')
    parser.add_argument('--data_root', default='../Datasets/', type=str, help='dataset root for cifar10, mnist, ..')
    parser.add_argument('--topk', default=30, type=int, help='number of sample per dataset in training loader')
    parser.add_argument('--dataset', default='joint', type=str, help='dataset choice amoung'
                                                                     ' [mnist, svhn, cifar10, stl10, joint')
    parser.add_argument('--split', default='train', type=str, help='dataset split{ train, test, val]')
    parser.add_argument('--ae_type', default='ldm', type=str, help='auto encoder type [ldm, vqvae, simple]')
    parser.add_argument('--save_path', default='ae_checkpoints', type=str, help='checkpointys folders')
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
        # metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. base_config_kl.yaml"
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",

        # default="stage2/configs/norm_base_config.yaml",
        # default="stage2/configs/base_llama_small_block_.yaml",
        # default="stage2/configs/llama_ffn_base_chunk.yaml",
        # default="stage2/configs/base_llama_attn_config.yaml",
        #
        default="stage2/configs/pythia_base_config.yaml",

        # default="stage2/configs/fnn_base_config.yaml",
        # default="stage2/configs/mini_llama_norm_config.yaml",
        #
        #
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
        default=24,
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
def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


if __name__ == "__main__":
    # seed_everything(seed=1234)
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    parser = get_parser()
    # parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()
    nowname= opt.name+now
    configs = [OmegaConf.load(opt.base)]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    # print(config.model)
    # exit()
    model = instantiate_from_config(config.model)
    ds = instantiate_from_config(config.data)
    ds.prepare_data()
    ds.setup(stage='fit')
    print("#### Data #####")
    # print(f'dataset {ds.dataset}')
    checkpoint_callback = ModelCheckpoint(monitor='train/loss_simple',
                                          dirpath='ldm_checkpoints/',
                                          filename='checkpoint_pythia_160M_413_top_25_{epoch}_',
                                          every_n_epochs=1
                                          )

    #checkpoint_base_top_1_tf_llama3_8n

    #checkpoint_model_base_llama3_instruct_attn this one

    #checkpoint_vagosol_ffn_16_llama3_8n
    #################checkpoint_model_attn_31_VAGOsolutions_##########
    #checkpoint_model_layer_attnv2_llama3-1-8b_
    trainer = pl.Trainer(accelerator="gpu", devices=-1, min_epochs=1000,
                         max_epochs=100000, log_every_n_steps=1, callbacks=[checkpoint_callback])
    trainer.fit(model, ds)
