import os

import lightning as L
import lightning.pytorch.callbacks as cbs
import torch
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.loggers.csv_logs import CSVLogger
from transformers import AutoTokenizer

from model import LanguageModel
from module.modeling_switcher import SwitcherConfig, SwitcherModelForCausalLM

torch.set_float32_matmul_precision('medium')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

config = SwitcherConfig()
model = SwitcherModelForCausalLM(config).to(torch.bfloat16)
print(model)
# summary(model)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b', legacy=False)

model = LanguageModel(
    model=model,
    tokenizer=tokenizer,
    parquet_path='/data/csy/transformers/hub/datasets--skymizer--fineweb-edu-dedup-45B/parquets',
    seq_len=2048,
    batch_size=6
)

checkpoint_callback = cbs.ModelCheckpoint(
    dirpath="./checkpoints",
    # save_last='link',
    save_on_exception=False,
    every_n_train_steps=1365 * 2,    # save checkpoint every 0.5B tokens. 512M / (6 * 4 * 8 * 2K) = 1365
    save_top_k=-1,
)

# logger = CSVLogger(save_dir='./logs/', flush_logs_every_n_steps=50)
logger = WandbLogger(project="hoplm-0.4b", name="gdn-0.4b-mu")

prog_callback = cbs.RichProgressBar()

lr_monitor = cbs.LearningRateMonitor(logging_interval="step")

trainer = L.Trainer(
    max_epochs=1,
    accelerator="gpu",
    devices=[0, 1, 2, 3],
    strategy="ddp",
    precision="bf16-mixed",
    # gradient_clip_val=1.0,
    callbacks=[checkpoint_callback, prog_callback, lr_monitor],
    logger=logger,
    log_every_n_steps=20,
    # val_check_interval=10920,
    val_check_interval=5460,
    num_sanity_val_steps=-1,
    # accumulate_grad_batches=8,
    enable_model_summary=True,
)

trainer.fit(
    model=model,
    # ckpt_path='checkpoints/epoch=0-step=5460.ckpt'
)
