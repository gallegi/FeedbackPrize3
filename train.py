import os
import argparse
import importlib
import pandas as pd
import torch

from transformers import AutoTokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, CometLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

from configs.default_config import CFG
from src.general import seed_torch
from src.dataset import FBDataset
from src.model import LitFB3

parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--config', type=str, default='default_config',
                    help='config file to run an experiment')

args = parser.parse_args()

config_module = importlib.import_module(f'configs.{args.config}')
CFG = config_module.CFG

seed_torch(CFG.seed) # set initial seed

CFG.output_dir_name = CFG.model.replace('/', '_') + '_' + CFG.version_note
CFG.output_dir = os.path.join(CFG.model_dir, CFG.output_dir_name)

df = pd.read_csv(CFG.data_file)

tokenizer = AutoTokenizer.from_pretrained(CFG.model)
tokenizer.save_pretrained(CFG.output_dir+'/tokenizer/')
CFG.tokenizer = tokenizer

os.makedirs(CFG.output_dir, exist_ok=True)

for valid_fold in CFG.run_folds:
    print(f'================= Training fold {valid_fold} ================')
    seed_torch(CFG.seed) # set seed each time a fold is run

    train_df = df[df['fold']!=valid_fold].reset_index(drop=True)
    valid_df = df[df['fold']==valid_fold].reset_index(drop=True)

    if(CFG.sample):
        train_df = train_df.sample(CFG.sample).reset_index(drop=True)
        valid_df = valid_df.sample(CFG.sample).reset_index(drop=True)

    CFG.num_train_examples = len(train_df)

    # Defining DataSet
    train_dataset = FBDataset(CFG, train_df)
    valid_dataset = FBDataset(CFG, valid_df)

    batch_size = CFG.batch_size
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True,
                                               pin_memory=True,drop_last=True,num_workers=CFG.num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=batch_size,num_workers=CFG.num_workers,
                                               shuffle=False,pin_memory=True,drop_last=False)

    CFG.steps_per_epoch = int(len(train_df) / CFG.batch_size)
    CFG.num_train_steps = int(len(train_df) / CFG.batch_size * CFG.epochs)
    lit_module = LitFB3(CFG, pretrained=True)

    # logger = CSVLogger(CFG.output_dir, name=f'fold{valid_fold}')
    logger = CometLogger(api_key=CFG.comet_api_key, project_name=CFG.comet_project_name, experiment_name=CFG.output_dir_name)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpointer = ModelCheckpoint(
         dirpath=os.path.join(CFG.output_dir, f'fold{valid_fold}'),
         filename='{epoch}-{val_loss:.3f}-{val_mean_MCRMSE:.3f}',
         monitor='val_loss',
         verbose=True,
         save_weights_only=True
    )
    trainer = Trainer(default_root_dir=CFG.output_dir, precision=16, max_epochs=CFG.epochs,
                     check_val_every_n_epoch=1, enable_checkpointing=True,
                     log_every_n_steps=CFG.steps_per_epoch,
                     logger=logger,
                     callbacks=[lr_monitor, checkpointer],
                     accelerator=CFG.accelerator, devices=CFG.devices)
    trainer.fit(lit_module, train_loader, valid_loader)
    # break