import lightning as L
import lightning.pytorch.callbacks as cbs
from data.stream_parquet import StreamingParquet
from torch.utils.data import DataLoader

import lm_eval
from lm_eval.tasks import TaskManager
from lm_eval.models.huggingface import HFLM

import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, ChainedScheduler, SequentialLR
from typing import Optional

def create_warmup_cosine_scheduler(
    optimizer,
    warmup_epochs: int,
    total_epochs: int,
    eta_min: float = 0.1,
    initial_lr: Optional[float] = None,
    last_epoch: int = -1
):
    """
    创建一个 '线性预热 + 余弦退火' 的组合学习率调度器（无重启）。
    
    参数:
        optimizer (torch.optim.Optimizer): 优化器实例
        warmup_epochs (int): 预热阶段的 epoch 数量
        total_epochs (int): 总训练轮数（余弦周期长度）
        eta_min (float): 余弦退火的最小学习率，默认 1e-6
        initial_lr (float, optional): 初始学习率。若为 None，则自动从 optimizer 获取
        last_epoch (int): 上一个 epoch 的索引，用于恢复训练，默认 -1
    
    返回:
        torch.optim.lr_scheduler.ChainedScheduler: 按顺序应用的调度器链
    """
    
    if initial_lr is None:
        initial_lr = optimizer.param_groups[0]['lr']
    
    # 第一阶段：线性预热（从 0.1 * initial_lr 线性增长到 initial_lr）
    scheduler_warmup = LinearLR(
        optimizer,
        start_factor=0.1,           # 起始为 10% 的初始学习率
        end_factor=1.0,             # 结束为 100%
        total_iters=warmup_epochs,  # 预热总步数
        last_epoch=last_epoch       # 支持断点恢复
    )
    
    # 第二阶段：余弦退火（从 pre-warmup 结束后的 lr 开始，衰减到 eta_min）
    # 注意：余弦周期长度 = total_epochs - warmup_epochs
    cosine_duration = max(1, total_epochs - warmup_epochs)  # 至少为1，避免除零
    scheduler_cosine = CosineAnnealingLR(
        optimizer,
        T_max=cosine_duration,      # 余弦周期长度
        eta_min=eta_min,
        last_epoch=last_epoch - warmup_epochs if last_epoch >= warmup_epochs else -1
    )
    
    # 使用 ChainedScheduler 按顺序串联两个调度器
    # 注意：ChainedScheduler 会依次调用每个 scheduler.step()
    scheduler = SequentialLR(
        optimizer=optimizer, 
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[warmup_epochs],
        last_epoch=last_epoch
    )
    
    return scheduler

class LanguageModel(L.LightningModule):
    def __init__(self, model, tokenizer, parquet_path, seq_len, batch_size):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.parquet_path = parquet_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.tokenizer.pad_token = self.tokenizer.eos_token


        # resume training dataset
        self.pq_idx = 0
        self.rg_idx = None

        self.automatic_optimization = False
        self.grad_accum = 8
        self.grad_clip = 1.0
        self.stream_loss = 0

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            ddp_rank = self.global_rank
            world_size = self.trainer.world_size
            self.train_dataset = StreamingParquet(self.parquet_path, self.batch_size, self.seq_len, self.tokenizer, ddp_rank=ddp_rank, world_size=world_size, split='train')
            # self.test_dataset = StreamingParquet(self.parquet_path, self.batch_size, self.seq_len, self.tokenizer, ddp_rank=ddp_rank, world_size=world_size, split='test')
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1)    # distributed sampling and batching done within StreamingParquet

    def val_dataloader(self):
         # return DataLoader(self.test_dataset, batch_size=1)    # dummy
         return [1]

    def training_step(self, batch, batch_idx):
        x, y, state_dict = batch
        x = x.squeeze(0)
        y = y.squeeze(0)
        self.pq_idx = state_dict['pq_idx']
        self.rg_idx = state_dict['rg_idx']
        output = self.model(input_ids=x, label=y)
        loss = output.loss

        # manual optimization for multiple optimizers
        loss = loss / self.grad_accum
        self.stream_loss += loss.item()
        self.manual_backward(loss)
        if (batch_idx + 1) % self.grad_accum == 0:
            for opt in self.optimizers():
                self.clip_gradients(opt, gradient_clip_val=self.grad_clip, gradient_clip_algorithm='norm')
                opt.step()
                opt.zero_grad()
            for sch in self.lr_schedulers():
                sch.step()
            self.log('train/loss', self.stream_loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
            self.stream_loss = 0
        
        # self.log('train/loss', loss.item(), on_step=True, prog_bar=True, logger=True)
        self.log('train/pq_idx', self.pq_idx.item(), on_step=True, prog_bar=True, logger=False)
        self.log('train/rg_idx', self.rg_idx.item(), on_step=True, prog_bar=True, logger=False)
        return loss
   

    def validation_step(self, batch, idx):
        pass

    def on_validation_epoch_end(self):
        self.model.device = torch.device(f'cuda:{self.global_rank}')
        model = HFLM(
            pretrained=self.model,
            tokenizer=self.tokenizer,
            backend='causal',
            add_bos_token=True
        )
        task_manager = TaskManager(
            metadata={
                "max_seq_lengths": [1024, 2048, 4096, 8192],
                "tokenizer": self.tokenizer,
                "shuffle": True,
                "enable_cache": True,
                "num_samples": 500,
            },
        )
        results = lm_eval.simple_evaluate(
            model=model,
            task_manager=task_manager,
            tasks=['lambada_openai'],
            batch_size=4,
            apply_chat_template=False
        )        
        self.print(results['results'])
        self.log('lambada_openai/perplexity', results['results']['lambada_openai']['perplexity,none'], logger=True, sync_dist=True)
        self.log('lambada_openai/acc', results['results']['lambada_openai']['acc,none'], logger=True, sync_dist=True)
        # self.log('arc/easy', results['results']['arc_easy']['acc,none'], logger=True, sync_dist=True)
        
    def on_save_checkpoint(self, checkpoint):
        pq_idx = torch.tensor([self.pq_idx], device=self.device)        
        rg_idx = torch.tensor([self.rg_idx], device=self.device)
        pq_idx = self.all_gather(pq_idx)
        rg_idx = self.all_gather(rg_idx)
        if self.global_rank == 0:
            checkpoint['dataset_state_dict'] = {'pq_idx': pq_idx, 'rg_idx': rg_idx}

    def on_load_checkpoint(self, checkpoint):
        pq_idx = checkpoint['dataset_state_dict']['pq_idx']
        rg_idx = checkpoint['dataset_state_dict']['rg_idx']
        self.pq_idx = pq_idx[self.global_rank].item()
        self.rg_idx = rg_idx[self.global_rank].item()
        self.print(f'resume to dataset at pq_idx = {self.pq_idx}, rg_idx = {self.rg_idx}')
        state_dict = {'pq_idx': pq_idx[self.global_rank].item(), 'rg_idx': rg_idx[self.global_rank].item()}
        self.train_dataset.load_state_dict(state_dict)
        # self.tune_optimizer(checkpoint)

    def tune_optimizer(self, state_dict):
        state_dict['lr_schedulers'][0]['_schedulers'][1]['T_max'] = 53893
        state_dict['lr_schedulers'][1]['_schedulers'][1]['T_max'] = 53893
        
    def configure_optimizers(self):
        emb_params = list(self.model.model.emb.parameters())
        hidden_1d_params = [p for n, p in self.model.model.enc.named_parameters() if p.dim() < 2 or p.dim() == 3]    # 3 for causal convolution parameters
        hidden_2d_params = [p for n, p in self.model.model.enc.named_parameters() if p.dim() == 2]
        head_params = list(self.model.lm_head.parameters())
        adam_groups = [
            dict(params=emb_params, lr=0.2),
            dict(params=head_params, lr=0.002),
            dict(params=hidden_1d_params, lr=0.02)
        ]
        adam_opt = torch.optim.AdamW(adam_groups, betas=(0.8, 0.95), weight_decay=0.1)
        muon_opt = torch.optim.Muon(hidden_2d_params, lr=0.02, momentum=0.95, weight_decay=0.1)
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=1e-3)
        adam_scheduler = create_warmup_cosine_scheduler(
            # optimizer=adam_opt, warmup_epochs=270, total_epochs=54163, eta_min=2e-4
            optimizer=adam_opt, warmup_epochs=270, total_epochs=54163, eta_min=2e-4
        )
        muon_scheduler = create_warmup_cosine_scheduler(
            optimizer=muon_opt, warmup_epochs=270, total_epochs=54163, eta_min=2e-3
        )
        adam_scheduler_cfg = {
            'scheduler': adam_scheduler,
            'interval': "step",
            'frequency': 1
        }
        muon_scheduler_cfg = {
            'scheduler': muon_scheduler,
            'interval': "step",
            'frequency': 1
        }
        # return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
        return [adam_opt, muon_opt], [adam_scheduler_cfg, muon_scheduler_cfg]
