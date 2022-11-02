import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW

from transformers import AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

import pytorch_lightning as pl
from .metric import get_score
# ====================================================
# Model
# ====================================================

def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if "model" not in n],
         'lr': decoder_lr, 'weight_decay': 0.0}
    ]
    return optimizer_parameters

    
def get_scheduler(cfg, optimizer, num_train_steps):
    if cfg.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
        )
    elif cfg.scheduler == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
        )
    return scheduler


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    
class LitFB3(pl.LightningModule):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.model_config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
            self.model_config.hidden_dropout = 0.
            self.model_config.hidden_dropout_prob = 0.
            self.model_config.attention_dropout = 0.
            self.model_config.attention_probs_dropout_prob = 0.
        else:
            self.model_config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.model_config)
        else:
            self.model = AutoModel(self.model_config)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.pool = MeanPooling()
        self.fc = nn.Linear(self.model_config.hidden_size, 6)
        self._init_weights(self.fc)

        self.criterion = nn.SmoothL1Loss(reduction='mean') # RMSELoss(reduction="mean")
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        # elif isinstance(module, nn.Embedding):
        #     module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
        #     if module.padding_idx is not None:
        #         module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        feature = self.pool(last_hidden_states, inputs['attention_mask'])
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(feature)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        optimizer_parameters = get_optimizer_params(self,
                                        encoder_lr=self.cfg.encoder_lr, 
                                        decoder_lr=self.cfg.decoder_lr,
                                        weight_decay=self.cfg.weight_decay)
        optimizer = AdamW(optimizer_parameters, lr=self.cfg.encoder_lr, eps=self.cfg.eps, betas=self.cfg.betas)
        num_train_steps = int(self.cfg.num_train_examples / self.cfg.batch_size * self.cfg.epochs)

        #Defining LR SCheduler
        lr_scheduler = get_scheduler(self.cfg, optimizer, num_train_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                # "monitor": "metric_to_track",
                # "frequency": "indicates how often the metric is updated"
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }


    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
            
        labels = labels.to(self.device)
        batch_size = labels.size(0)
        preds = self(inputs)
        loss = self.criterion(preds, labels)
        self.log('train_loss', loss)
        return {'loss': loss, 'preds':preds, 'labels':labels}

    def validation_step(self, val_batch, batch_idx):
        inputs, labels = val_batch
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
            
        labels = labels.to(self.device)
        batch_size = labels.size(0)
        preds = self(inputs)
        loss = self.criterion(preds, labels)
        self.log('val_loss', loss)

        return {'loss': loss, 'preds':preds, 'labels':labels}

    def training_epoch_end(self, training_step_outputs):
        all_preds = np.concatenate([outputs['preds'].detach().cpu().numpy() for outputs in training_step_outputs])
        all_labels = np.concatenate([outputs['labels'].detach().cpu().numpy() for outputs in training_step_outputs])

        mean_MCRMSE, MCRMSEs = get_score(y_trues=all_labels, y_preds=all_preds)
        self.log('train_mean_MCRMSE', mean_MCRMSE)
        for col, metric_val in zip(self.cfg.target_cols, MCRMSEs):
            self.log('train_' + col + '_MCRMSE', metric_val)

    def validation_epoch_end(self, validation_step_outputs):
        all_preds = np.concatenate([outputs['preds'].detach().cpu().numpy() for outputs in validation_step_outputs])
        all_labels = np.concatenate([outputs['labels'].detach().cpu().numpy() for outputs in validation_step_outputs])

        mean_MCRMSE, MCRMSEs = get_score(y_trues=all_labels, y_preds=all_preds)
        self.log('val_mean_MCRMSE', mean_MCRMSE)
        for col, metric_val in zip(self.cfg.target_cols, MCRMSEs):
            self.log('val_' + col + '_MCRMSE', metric_val)