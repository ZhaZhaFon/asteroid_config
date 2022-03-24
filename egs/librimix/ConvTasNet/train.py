
# original codebase: https://github.com/asteroid-team/asteroid
# modified and re-distributed by Zifeng Zhao @ Peking University
# 2022.03

import os
import argparse
import json

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from asteroid.models import ConvTasNet
from asteroid.data import LibriMix
from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.system import System
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")
# 新增
parser.add_argument("--conf", type=str, required=True, help='Path to the config(.yaml) file')
parser.add_argument("--resume", type=str, help='Folder where the checkpoints were saved')
parser.add_argument("--debug", type=bool, help='debug mode')

def main(conf):
    
    print('>> 读取数据...')
    print('   >> 读取train_set...')
    train_set = LibriMix(
        csv_dir=conf["data"]["train_dir"],
        task=conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        n_src=conf["data"]["n_src"],
        segment=conf["data"]["segment"],
    )
    print('   >> 读取train_set...')
    val_set = LibriMix(
        csv_dir=conf["data"]["valid_dir"],
        task=conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        n_src=conf["data"]["n_src"],
        segment=conf["data"]["segment"],
    )

    print('>> 加载DataLoader...')
    print('   >> 加载train_loader...')
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )
    print('   >> 加载val_set...')
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )
    conf["masknet"].update({"n_src": conf["data"]["n_src"]})

    print('>> 建立model')
    model = ConvTasNet(
        **conf["filterbank"], **conf["masknet"], sample_rate=conf["data"]["sample_rate"]
    )
    print('>> 加载optimizer/scheduler')
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    # Define scheduler
    scheduler = None
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)
    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]
    print(f'>> 实验保存到{exp_dir}')
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, conf['main_args']['conf'].split('/')[-1])
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    print(f'>> 建立system')
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    system = System(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=1, verbose=True
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=30, verbose=True))

    # Don't ask GPU if they are not available.
    gpus = -1 if torch.cuda.is_available() else None
    distributed_backend = "ddp" if torch.cuda.is_available() else None

    resume_from_checkpoint = None
    if conf['main_args']['resume']:
        resume_from_checkpoint = os.path.join(conf['main_args']['resume'], 'last.ckpt')
        print(f'>> 加载断点 {resume_from_checkpoint}...')
    max_epoch = conf["training"]["epochs"]
    if conf['main_args']['debug']:
        max_epoch = 5
        
    print('>> 建立trainer...')
    trainer = pl.Trainer(
        max_epochs=max_epoch,
        callbacks=callbacks,
        checkpoint_callback=checkpoint,
        default_root_dir=exp_dir,
        gpus=gpus,
        distributed_backend=distributed_backend,
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    
    print('>> 启动trianer.fit()...')
    print('')
    print('### START TRAINING ###')
    print('')
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    print(f'>> trainer.fit()完成 best_model保存到{os.path.join(exp_dir, "best_model.pth")}')
    to_save = system.model.serialize()
    to_save.update(train_set.get_infos())
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))
    
    print('')
    print('### TRAINING COMPLETED ###')
    print('')
    from IPython import embed
    embed()
    
    torch.cuda.empty_cache()
    print('')
    print('done.')

if __name__ == "__main__":
    import yaml
    from pprint import pprint
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    args = parser.parse_args()
    print('')
    print(f'>> 解析超参数 {parser}')
    with open(args.conf) as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    pprint(arg_dic)
    main(arg_dic)
