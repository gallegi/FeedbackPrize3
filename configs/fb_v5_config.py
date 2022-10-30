# weighted sum of multiple hidden layers
class CFG:
    root_folder = './'
    run_folds = [0,1,2,3,4]
    accelerator = 'gpu'
    devices = 1
    comet_api_key = 'zR96oNVqYeTUXArmgZBc7J9Jp'
    comet_project_name = 'FeedbackPrize3'
    num_workers=0
    model="microsoft/deberta-v3-base"
    n_hidden_pool = 4
    gradient_checkpointing=True
    scheduler='cosine' # ['linear', 'cosine']
    batch_scheduler=True
    num_cycles=0.5
    num_warmup_steps=0
    epochs=10
    encoder_lr=2e-5
    decoder_lr=2e-5
    min_lr=1e-6
    eps=1e-6
    betas=(0.9, 0.999)
    batch_size=16
    max_len=1429
    weight_decay=0.01
    gradient_accumulation_steps=1
    max_grad_norm=1000
    target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed=42
    n_fold=45
    version_note = 'lit_v4_comet'
    sample = None
    patience = 10

CFG.data_file = f'{CFG.root_folder}/data/train_5folds.csv'
CFG.model_dir = f'{CFG.root_folder}/models/'