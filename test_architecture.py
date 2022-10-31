# %%
import sys
sys.path.append('../')

# %%
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

from configs.default_config import CFG
from src.dataset import FBDataset
from src.model import LitFB3

seed_torch(CFG.seed) # set initial seed

CFG.output_dir_name = CFG.model.replace('/', '_') + '_' + CFG.version_note
CFG.output_dir = os.path.join(CFG.model_dir, CFG.output_dir_name)


# %%
df = pd.read_csv('/Users/namnguyenthe/Workspace/Kaggle/FeedbackPrize3/data/train_5folds.csv')

tokenizer = AutoTokenizer.from_pretrained(CFG.model)
tokenizer.save_pretrained(CFG.output_dir+'/tokenizer/')
CFG.tokenizer = tokenizer

# %%
ds = FBDataset(CFG, df)
dataloader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)

# %%
inputs, labels = next(iter(dataloader))

# %%
lit_module = LitFB3(CFG, pretrained=True)

# %%
lit_module(inputs)
