import random
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
from jaxtyping import Float, Int

from fla.models.delta_net.modeling_delta_net import DeltaNetBlock
from fla.models.delta_net.configuration_delta_net import DeltaNetConfig
# from fla.models.gated_deltanet.modeling_gated_deltanet import GatedDeltaNetBlock
# from fla.models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetConfig
from fla.modules.mlp import GatedMLP
from fla.layers.utils import get_unpad_data, index_first_axis
from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss

from einops import rearrange

from .switcher import CrossSwitcher, apply_rotary_pos_emb
from .cache_utils import Cache
# from .gdn import SlowGatedDeltaNetBlock

@dataclass
class SwitcherConfig:
    vocab_size: int = 32000    # [BUG] 151644 or 151643?
    n_layers: int = 24
    dim: int = 1024
    rotary_dim: int = 128
    n_heads: int = 8
    rotary_base: int = 10000
    expand_ratio: int = 4
    hybrid_freq: int = 1  # every k layers, use hybrid switcher attention. Otherwise use delta rule.
    random_force_attn: bool = True  # during training, force at least one decoder layer to use attention to prevent unused parameter error from DDP

@dataclass
class ModelOutput:
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[dict] = None
    loss: Optional[torch.Tensor] = None

class RotaryEmbedding(nn.Module):
    
    def __init__(self, config: SwitcherConfig):
        super().__init__()
        self.rotary_dim = config.rotary_dim
        self.rotary_base = config.rotary_base

        assert self.rotary_dim % 2 == 0

        inv_freq = 1.0 / (self.rotary_base ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)


    def forward(
        self, 
        hidden_state: Float[torch.Tensor, 'b t d'], 
        position_ids: Int[torch.Tensor, 'b t']
    ) -> Tuple[
        Float[torch.Tensor, 'b t rd'], 
        Float[torch.Tensor, 'b t rd']
    ]:
        self.inv_freq = self.inv_freq.to(position_ids)
        freqs = torch.einsum('b t, d->b t d', position_ids, self.inv_freq)
        freqs = torch.cat([freqs, freqs], dim=-1)
        cos = freqs.cos()
        sin = freqs.sin()
        return cos.to(hidden_state), sin.to(hidden_state)

class SwitcherLayer(nn.Module):

    def __init__(self, layer_idx: int, config: SwitcherConfig):
        super().__init__()
        self.layer_idx = layer_idx
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads

        self.input_norm = nn.RMSNorm(self.dim)
        self.post_norm = nn.RMSNorm(self.dim)

        self.attn = CrossSwitcher(self.layer_idx, self.dim, self.n_heads)
        self.mlp = GatedMLP(hidden_size=config.dim, hidden_ratio=config.expand_ratio)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        k_cache: torch.Tensor, 
        v_cache: torch.Tensor, 
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        force_attn: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        layer_cache: Optional[Cache] = None, 
        use_cache: bool = False,
    ):
        x = self.input_norm(hidden_states)
        x, s = self.attn(
            x, 
            k_cache=k_cache, 
            v_cache=v_cache,
            position_embeddings=position_embeddings,
            force_attn=force_attn,
            cu_seqlens=cu_seqlens,
            past_key_values=layer_cache, 
            use_cache=use_cache
        )
        hidden_states = hidden_states + x

        x = self.post_norm(hidden_states)
        x = self.mlp(x)
        hidden_states = hidden_states + x

        return hidden_states, layer_cache
        
class SwitcherModel(nn.Module):

    def __init__(self, config: SwitcherConfig):
        super().__init__()
        assert config.n_layers % 2 == 0
        self.config = config
        self.n_layers = config.n_layers
        self.n_enc_layers = config.n_layers // 2
        self.n_dec_layers = config.n_layers - self.n_enc_layers
        self.random_force_attn = config.random_force_attn
        assert self.n_dec_layers % config.hybrid_freq == 0

        self.switcher_idx = [i for i in range(self.n_dec_layers) if i % config.hybrid_freq == 0]

        delta_config = DeltaNetConfig(
            hidden_size=config.dim,
            expand_v=1,
            num_heads=config.n_heads,
            head_dim=config.dim // config.n_heads,
            hidden_ratio=config.expand_ratio
        )

        self.rotary_emb = RotaryEmbedding(config)

        self.emb = nn.Embedding(config.vocab_size, config.dim)
        self.enc = nn.ModuleList([
            DeltaNetBlock(config=delta_config, layer_idx=i) for i in range(self.n_enc_layers)
        ])
        self.cache_k_proj = nn.Linear(config.dim, config.dim)
        self.cache_v_proj = nn.Linear(config.dim, config.dim)
        self.dec = nn.ModuleList([
            SwitcherLayer(i, config) if i % config.hybrid_freq == 0 else DeltaNetBlock(config=delta_config, layer_idx=i) 
            for i in range(self.n_dec_layers)
        ])
        self.norm = nn.RMSNorm(config.dim)

    def forward(
        self, 
        input_ids: Int[torch.Tensor, 'b t'], 
        position_ids: Optional[Int[torch.Tensor, 'b t']] = None,      # decided by generate(). when decoding, will contain only decoded part
        attention_mask: Optional[Int[torch.Tensor, 'b t']] = None,    # decided by generate(). when decoding, will contain both prefill and generated part
        use_cache: bool = False, 
        past_key_values: Optional[Cache] = None
    ):

        layer_cache = None
        if use_cache and past_key_values is None:
            layer_cache = Cache()
            past_key_values = {
                'k_cache': None,
                'v_cache': None,
                'layer_cache': layer_cache,    # recurrent cache, for rnn state [B, H, d, d] * L and conv state [B, Hd, conv_d] * L 
            }

        hidden_states = self.emb(input_ids)
        
        cu_seqlens = None
        if attention_mask is not None:
            input_length = hidden_states.shape[1]
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -input_length:])    # padding is on the left during inference. Disabled in training.
            # hidden_states: [B, T, d] -> [1, (total_length), d]
            hidden_states = index_first_axis(rearrange(hidden_states, 'b s ... -> (b s) ...'), indices).unsqueeze(0)
            
        if position_ids is None:
            # past_seen_tokens = layer_cache.get_sequence_length()   # [NOTE] not really past tokens. used only for training to decide position ids
            if attention_mask is not None:
                position_ids = torch.cumsum(attention_mask, dim=1).long() - 1
                position_ids = index_first_axis(position_ids.flatten().unsqueeze(-1), indices).squeeze(-1).unsqueeze(0)
            else:
                position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).expand_as(input_ids)    # maximum length in the batch
        position_ids = position_ids.to(hidden_states.device)

        attention_mask = None    # only use cu_seqlens instead of attention_mask for varlen inference

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.enc:
            hidden_states, _, layer_cache = layer(hidden_states, attention_mask=attention_mask, past_key_values=layer_cache, use_cache=use_cache, cu_seqlens=cu_seqlens)
        
        k_cache = self.cache_k_proj(hidden_states)
        v_cache = self.cache_v_proj(hidden_states)
        cos, sin = position_embeddings
        k_cache = k_cache.view(hidden_states.shape[0], hidden_states.shape[1], self.config.n_heads, -1).transpose(1, 2)
        v_cache = v_cache.view(hidden_states.shape[0], hidden_states.shape[1], self.config.n_heads, -1).transpose(1, 2)
        k_cache = k_cache / torch.linalg.norm(k_cache, dim=-1, keepdim=True)
        k_cache = apply_rotary_pos_emb(k_cache, cos, sin, position_ids=position_ids)

        if use_cache and past_key_values is not None:
            if past_key_values['k_cache'] is None and past_key_values['v_cache'] is None:
                past_key_values['k_cache'] = k_cache
                past_key_values['v_cache'] = v_cache
            else:
                past_key_values['k_cache'] = torch.cat((past_key_values['k_cache'], k_cache), dim=1)
                past_key_values['v_cache'] = torch.cat((past_key_values['v_cache'], v_cache), dim=1)
            k_cache = past_key_values['k_cache']
            v_cache = past_key_values['v_cache']

        force_attn_idx = random.choice(self.switcher_idx) if self.random_force_attn else -1
        for layer in self.dec:
            if isinstance(layer, SwitcherLayer):
                hidden_states, layer_cache = layer(
                    hidden_states, 
                    k_cache=k_cache, 
                    v_cache=v_cache,
                    position_embeddings=position_embeddings,
                    force_attn=(layer.layer_idx == force_attn_idx),
                    cu_seqlens=cu_seqlens,
                    layer_cache=layer_cache,
                    use_cache=use_cache
                )
            else:
                hidden_states, _, layer_cache = layer(hidden_states, attention_mask=attention_mask, past_key_values=layer_cache, use_cache=use_cache, cu_seqlens=cu_seqlens)

        hidden_states = self.norm(hidden_states)
        # return ModelOutput(last_hidden_state=hidden_states, past_key_values=past_key_values)
        return hidden_states, past_key_values

class SwitcherModelForCausalLM(nn.Module):
    def __init__(self, config: SwitcherConfig):
        super().__init__()
        self.config = config
        self.model = SwitcherModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.criteria = FusedLinearCrossEntropyLoss()

    def forward(
        self, 
        input_ids: Int[torch.Tensor, 'b t'], 
        position_ids: Optional[Int[torch.Tensor, 'b t']] = None,      # decided by generate(). when decoding, will contain only decoded part
        attention_mask: Optional[Int[torch.Tensor, 'b t']] = None,    # decided by generate(). when decoding, will contain both prefill and generated part
        label: Optional[Int[torch.Tensor, 'b t']] = None,             # already shifted labels
        use_cache: bool = False, 
        past_key_values: Optional[Cache] = None
    ):
        outputs = self.model(
            input_ids, 
            position_ids=position_ids,
            attention_mask=attention_mask, 
            use_cache=use_cache, 
            past_key_values=past_key_values
        )
        hidden_states = outputs[0]
        past_key_values = outputs[1]
        
        logits = None
        if not self.training:
            logits = self.lm_head(hidden_states)
            
        loss = None
        if label is not None:
            # label = torch.cat((label[..., 1:], torch.full_like(label[:, :1], self.criterion.ignore_index)), dim=1)
            loss = self.criteria(hidden_states, label, self.lm_head.weight, self.lm_head.bias)
            
        return ModelOutput(
            logits=logits,
            past_key_values=past_key_values,
            loss=loss
        )

if __name__ == '__main__':
    config = SwitcherConfig()
    model = SwitcherModelForCausalLM(config).to(torch.bfloat16).cuda()
    input_ids = torch.randint(0, 10000, (2, 2048)).cuda()
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model(input_ids=input_ids, label=input_ids)
    loss = output.loss
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is None:
            print(name)    
