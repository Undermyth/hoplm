import torch
import lightning as L
import lightning.pytorch.callbacks as cbs
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.loggers.csv_logs import CSVLogger
from torchinfo import summary
from transformers import AutoTokenizer

from model import LanguageModel
from module.modeling_switcher import SwitcherConfig, SwitcherModelForCausalLM

torch.set_float32_matmul_precision('medium')

config = SwitcherConfig()
model = SwitcherModelForCausalLM(config).to(torch.bfloat16)
# summary(model)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b', legacy=False)

model = LanguageModel(
    model=model,
    tokenizer=tokenizer,
    parquet_path='/data2/csy/transformers/hub/datasets--skymizer--fineweb-edu-dedup-45B/parquets',
    seq_len=2048,
    batch_size=1
)

checkpoint_callback = cbs.ModelCheckpoint(
    dirpath="./checkpoints",
    # save_last='link',
    save_on_exception=False,
    every_n_train_steps=50,
    save_top_k=-1,
)

logger = CSVLogger(save_dir='./logs/', flush_logs_every_n_steps=50)
# logger = WandbLogger(project="hoplm-0.4b", name="hoplm-0.4b")

prog_callback = cbs.RichProgressBar()

lr_monitor = cbs.LearningRateMonitor(logging_interval="step")

trainer = L.Trainer(
    max_epochs=1,
    accelerator="gpu",
    devices=1,
    # strategy="ddp",
    strategy="ddp_find_unused_parameters_true",
    precision="bf16-mixed",
    callbacks=[checkpoint_callback, prog_callback, lr_monitor],
    logger=logger,
    log_every_n_steps=5,
    val_check_interval=400,
    num_sanity_val_steps=0,
    accumulate_grad_batches=8,
    enable_model_summary=True,
)

trainer.fit(
    model=model
)
