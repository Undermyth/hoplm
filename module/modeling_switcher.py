import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
from jaxtyping import Float, Int

from fla.layers import DeltaNet

from fla.layers.utils import get_unpad_data, index_first_axis
from einops import rearrange

from .switcher import CrossSwitcher
from .cache_utils import Cache

@dataclass
class SwitcherConfig:
    vocab_size: int
    n_layers: int
    dim: int
    rotary_dim: int
    n_heads: int
    rotary_emb_dim: int
    rotary_base: int
    expand_ratio: int
    hybrid_freq: int   # every k layers, use hybrid switcher attention. Otherwise use delta rule.

@dataclass
class ModelOutput:
    last_hidden_state: torch.FloatTensor
    past_key_values: Optional[dict] = None

class RotaryEmbedding(nn.Module):
    
    def __init__(self, config: SwitcherConfig):
        super().__init__()
        self.rotary_dim = config.rotary_dim
        self.rotary_base = config.rotary_base

        assert self.rotary_dim // 2 == 0

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
        freqs = torch.einsum('b t, d->b t d', position_ids, self.inv_freq)
        freqs = torch.cat([freqs, freqs], dim=-1)
        cos = freqs.cos()
        sin = freqs.sin()
        return cos.to(hidden_state.dtype), sin.to(hidden_state.dtype)

class MLP(nn.Module):

    def __init__(self, dim, expand_ratio):
        super().__init__()
        self.dim = dim
        hidden_dim = int(dim * expand_ratio)
        self.up_proj = nn.Linear(dim, hidden_dim)
        self.gate_proj = nn.Linear(dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, dim)
        self.act_fn = nn.Sigmoid()

    def forward(self, x):
        x = self.up_proj(x)
        gate = self.act_fn(self.gate_proj(x))
        return self.down_proj(x * gate)

class SwitcherLayer(nn.Module):

    def __init__(self, config: SwitcherConfig):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.rotary_emb_dim = config.rotary_emb_dim

        self.input_norm = nn.RMSNorm(self.dim)
        self.post_norm = nn.RMSNorm(self.dim)

        self.attn = CrossSwitcher(self.dim, self.n_heads, self.rotary_emb_dim)
        self.mlp = MLP(self.dim, config.expand_ratio)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        k_cache: torch.Tensor, 
        v_cache: torch.Tensor, 
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: Optional[torch.Tensor] = None,
        layer_cache: Optional[Cache] = None, 
        use_cache: bool = False,
    ):
        x = self.input_norm(hidden_states)
        x = self.attn(
            x, 
            k_cache=k_cache, 
            v_cache=v_cache,
            position_embeddings=position_embeddings,
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
        assert config.n_layers // 2 == 0
        self.n_layers = config.n_layers
        self.n_enc_layers = config.n_layers // 2
        self.n_dec_layers = config.n_layers - self.n_enc_layers
        assert self.n_dec_layers // config.hybrid_freq == 0

        self.rotary_emb = RotaryEmbedding(config)

        self.emb = nn.Embedding(config.vocab_size, config.dim)
        self.enc = nn.ModuleList([
            DeltaNet(hidden_size=config.dim, num_heads=config.n_heads) for _ in range(self.n_enc_layers)
        ])
        self.cache_k_proj = nn.Linear(config.dim, config.dim)
        self.cache_v_proj = nn.Linear(config.dim, config.dim)
        self.dec = nn.ModuleList([
            SwitcherLayer(config) if i % config.hybrid_freq == 0 else DeltaNet(hidden_size=config.dim, num_heads=config.n_heads) 
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
            input_length = input_ids.shape[-1]
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -input_length:])    # padding is on the left during inference. Disabled in training.
            # hidden_states: [B, T, d] -> [1, (total_length), d]
            hidden_states = index_first_axis(rearrange(input_ids, 'b s ... -> (b s) ...'), indices).unsqueeze(0)
            attention_mask = None    # only use cu_seqlens instead of attention_mask for varlen inference
            
        if position_ids is None:
            # past_seen_tokens = layer_cache.get_sequence_length()   # [NOTE] not really past tokens. used only for training to decide position ids
            position_ids = torch.arange(0, hidden_states.shape[1], device=hidden_states.device)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.enc:
            hidden_states, _, layer_cache = layer(hidden_states, attention_mask=attention_mask, past_key_values=layer_cache, use_cache=use_cache, cu_seqlens=cu_seqlens)
        
        k_cache = self.cache_k_proj(hidden_states)
        v_cache = self.cache_v_proj(hidden_states)

        if use_cache and past_key_values is not None:
            if past_key_values['k_cache'] is None and past_key_values['v_cache'] is None:
                past_key_values['k_cache'] = k_cache
                past_key_values['v_cache'] = v_cache
            else:
                past_key_values['k_cache'] = torch.cat((past_key_values['k_cache'], k_cache), dim=1)
                past_key_values['v_cache'] = torch.cat((past_key_values['v_cache'], v_cache), dim=1)
            k_cache = past_key_values['k_cache']
            v_cache = past_key_values['v_cache']

        for layer in self.dec:
            if isinstance(layer, SwitcherLayer):
                hidden_states, layer_cache = layer(
                    hidden_states, 
                    k_cache=k_cache, 
                    v_cache=v_cache,
                    position_embeddings=position_embeddings,
                    cu_seqlens=cu_seqlens,
                    past_key_values=layer_cache,
                    use_cache=use_cache
                )
            else:
                hidden_states, _, layer_cache = layer(hidden_states, attention_mask=attention_mask, past_key_values=layer_cache, use_cache=use_cache, cu_seqlens=cu_seqlens)

        hidden_states = self.norm(hidden_states)
        return ModelOutput(last_hidden_state=hidden_states, past_key_values=past_key_values)

class SwitcherModelForCausalLM(nn.Module):
    def __init__(self, config: SwitcherConfig):
        super().__init__()
        self.model = SwitcherModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False, past_key_values=None):
        outputs = self.model(input_ids, attention_mask, use_cache, past_key_values)
        logits = outputs.last_hidden_state
        logits = self.lm_head(logits)
        return logits
