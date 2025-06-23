#  Copyright (c) 2024-2025, Christian Gilli <christian.gilli11@gmail.com>
#  All rights reserved.
#
#  This code is licensed under the terms of the 3-clause BSD license
#  (https://opensource.org/license/bsd-3-clause).


import dsc
import dsc.nn as nn
import dsc.nn.functional as F
from dataclasses import dataclass
from time import perf_counter
import argparse
from transformers import AutoTokenizer
from typing import Tuple, Optional, List
import math
import numpy as np


CacheEntry = Tuple[dsc.Tensor, dsc.Tensor]
Cache = List[CacheEntry]


# Default config for Qwen 2.5 0.5B
@dataclass
class Config:
    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    max_position_embeddings: int = 1024
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    rope_theta: float = 1000000.0
    sliding_window: int = 4096
    max_window_layers: int = 28
    bos_token_id: int = 151643
    eos_token_id: int = 151645


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    @dsc.trace('MLP')
    def forward(self, x: dsc.Tensor) -> dsc.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def _pre_compute_freqs(dim: int, theta: float, max_seq_len: int) -> Tuple[dsc.Tensor, dsc.Tensor]:
    freqs = 1.0 / (theta ** ((dsc.arange(start=0, stop=dim, step=2)[: (dim // 2)]).cast(dsc.f32) / dim))
    t = dsc.arange(stop=max_seq_len, dtype=dsc.f32)
    freqs = dsc.outer(t, freqs)
    cos_cache_half = dsc.cos(freqs)
    sin_cache_half = dsc.sin(freqs)

    cos_cache = dsc.concat([cos_cache_half, cos_cache_half], axis=-1)
    sin_cache = dsc.concat([sin_cache_half, sin_cache_half], axis=-1)
    return cos_cache, sin_cache


def _rotate_half(x: dsc.Tensor) -> dsc.Tensor:
    lim = x.size(-1) // 2
    x1 = x[:, :, :, :lim]
    x2 = x[:, :, :, lim:]
    return dsc.concat([-x2, x1], axis=-1)


@dsc.trace('RoPE')
def _apply_rope(q: dsc.Tensor, k: dsc.Tensor,
                freq_cos: dsc.Tensor,
                freq_sin: dsc.Tensor,
                position_ids: dsc.Tensor) -> Tuple[dsc.Tensor, dsc.Tensor]:
    cos = freq_cos[position_ids]
    sin = freq_sin[position_ids]

    batch_size, seq_len, head_size = cos.shape

    cos = cos.reshape(batch_size, 1, seq_len, head_size)
    sin = sin.reshape(batch_size, 1, seq_len, head_size)

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(x: dsc.Tensor, n_rep: int) -> dsc.Tensor:
    if n_rep == 1:
        return x
    return dsc.repeat(x, n_rep, axis=1)


class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.head_size = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.sliding_window = config.sliding_window
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_size)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_size)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_size)
        self.o_proj = nn.Linear(self.num_heads * self.head_size, config.hidden_size, bias=False)

    @dsc.trace('Attention')
    def forward(
        self, x: dsc.Tensor,
        freq_cos_cache: dsc.Tensor,
        freq_sin_cache: dsc.Tensor,
        position_ids: dsc.Tensor,
        past_key_value: Optional[CacheEntry] = None
    ) -> Tuple[dsc.Tensor, CacheEntry]:

        block_size, seq_len, _ = x.shape
        q, k_cur, v_cur = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        q = q.reshape(block_size, seq_len, self.num_heads, self.head_size).transpose((0, 2, 1, 3))
        k_cur = k_cur.reshape(block_size, seq_len, self.num_kv_heads, self.head_size).transpose((0, 2, 1, 3))
        v_cur = v_cur.reshape(block_size, seq_len, self.num_kv_heads, self.head_size).transpose((0, 2, 3, 1))

        q, k_cur = _apply_rope(q, k_cur, freq_cos_cache, freq_sin_cache, position_ids)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = dsc.concat([past_k, k_cur], axis=2)
            v = dsc.concat([past_v, v_cur], axis=3)
        else:
            k = k_cur
            v = v_cur

        present_key_value = (k, v)

        k = _repeat_kv(k, self.n_rep)
        v = _repeat_kv(v, self.n_rep)

        scores = dsc.matmul(q, k, trans_b=True) * (1.0 / math.sqrt(self.head_size))

        q_len = q.size(2)
        k_len = k.size(2)

        # SWA
        k_pos_indices = dsc.arange(k_len).reshape(1, -1)
        q_pos_indices = dsc.arange(start=(k_len - q_len), stop=k_len).reshape(-1, 1)
        causal_mask = k_pos_indices <= q_pos_indices # shape (q_len, k_len)
        window_mask = (q_pos_indices - k_pos_indices) < self.sliding_window

        should_attend = causal_mask * window_mask # shape (q_len, k_len)

        additive_mask = dsc.where(
            should_attend,
            0.0,
            float('-inf')
        ).reshape(1, 1, q_len, k_len)

        masked_scores = scores + additive_mask

        attn_weights = F.softmax(masked_scores, axis=-1)
        out = dsc.matmul(attn_weights, v, trans_b=True).transpose((0, 2, 1, 3)).reshape(block_size, seq_len, -1)

        return self.o_proj(out), present_key_value


class DecoderLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

    @dsc.trace('DecoderLayer')
    def forward(
        self,
        x: dsc.Tensor,
        freq_cos: dsc.Tensor,
        freq_sin: dsc.Tensor,
        position_ids: dsc.Tensor,
        past_key_value: Optional[CacheEntry] = None
    ) -> Tuple[dsc.Tensor, CacheEntry]:

        ln_out = self.input_layernorm(x)
        attn_out, present_kv = self.self_attn(ln_out, freq_cos, freq_sin, position_ids, past_key_value)
        h = x + attn_out
        return h + self.mlp(self.post_attention_layernorm(h)), present_kv


class Qwen25Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        cos_cache, sin_cache = _pre_compute_freqs(config.hidden_size // config.num_attention_heads, config.rope_theta, config.max_position_embeddings)
        self.cos_cache = cos_cache
        self.sin_cache = sin_cache


    @staticmethod
    def from_pretrained(config: Config = Config()) -> 'Qwen25Model':
        state_dict = nn.safe_load('https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct/resolve/main/model.safetensors',
                                  trim_prefix='model.',
                                  use_dtype=dsc.f32)
        model = Qwen25Model(config)
        model.from_state(state_dict,
                         tied={'lm_head.weight': 'embed_tokens.weight'})
        del state_dict
        dsc.print_mem_usage()
        return model
    
    @dsc.trace('Qwen2_5')
    def forward(self,
        x: dsc.Tensor,
        position_ids: dsc.Tensor,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = True
    ) -> Tuple[dsc.Tensor, Optional[Cache]]:
        h = self.embed_tokens(x.to('cpu'))

        next_kv_caches = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            layer_cache = past_key_values[i] if past_key_values is not None else None
            h, present_kv = layer(
                h,
                self.cos_cache,
                self.sin_cache,
                position_ids=position_ids,
                past_key_value=layer_cache
            )
            if use_cache:
                next_kv_caches.append(present_kv)

        h = self.norm(h)
        return self.lm_head(h), next_kv_caches

    @dsc.profile()
    def generate(self, idx: dsc.Tensor, tokenizer, max_new_tokens: int, top_k: int = 10):
        prompt_processing_start = perf_counter()
        prompt_tokens = idx.reshape(1, -1)
        prompt_len = prompt_tokens.size(1)

        prompt_position_ids = dsc.arange(stop=prompt_len, device='cpu').reshape(1, -1)
        # Run forward without caching
        logits, past_key_values = self(prompt_tokens, position_ids=prompt_position_ids, past_key_values=None)
        next_token_logits = logits[:, -1, :]
        generated_tokens = []
        current_len = prompt_len
        prompt_processing_ms = (perf_counter() - prompt_processing_start) * 1e3

        # Loop
        generation_start = perf_counter()
        for _ in range(max_new_tokens):
            v, _ = dsc.topk(next_token_logits, top_k)
            k_th_value = v[:, -1]
            next_token_logits = next_token_logits.masked_fill(next_token_logits < k_th_value, float('-inf'))
            probs = F.softmax(next_token_logits, axis=-1)

            next_token_id = dsc.multinomial(probs, num_samples=1)
            tok_id_scalar = next_token_id[0, 0]
            if tok_id_scalar == self.config.eos_token_id:
                print('\n[EOS]', flush=True)
                break

            generated_tokens.append(tok_id_scalar)
            print(tokenizer.decode(tok_id_scalar, skip_special_tokens=True), end='', flush=True)

            input_ids = next_token_id
            position_ids = dsc.tensor([current_len], dtype=dsc.i32, device='cpu').reshape(1, -1)

            # Run forward with caching
            logits, next_past_key_values = self(
                input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values
            )
            past_key_values = next_past_key_values
            next_token_logits = logits[:, -1, :] # Note: this is probably useless
            current_len += 1

        generation_stop = perf_counter()
        total_processing_ms = (generation_stop - prompt_processing_start) * 1e3
        generation_processing_ms = (generation_stop - generation_start) * 1e3
        print()

        print(f'prompt processing time\t= {round(prompt_processing_ms, 1)}ms')
        print(f'generation time\t\t= {round(generation_processing_ms, 1)} ms | {round(generation_processing_ms / max_new_tokens, 2)} ms/tok')
        print(f'total time\t\t= {round(total_processing_ms, 1)} ms | {round(max_new_tokens / (total_processing_ms / 1e3), 2)} tok/s')
        return generated_tokens


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description='QWEN 2.5 inference CLI')
    cli.add_argument('prompt', type=str, help='Model prompt')
    cli.add_argument('-n', type=int, default=100, help='Tokens to generate (default=100)')
    cli.add_argument('-top-k', type=int, default=10, help='Top K sampling (default=10)')
    cli.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Device on which to run the model')

    args = cli.parse_args()

    dsc.set_default_device(args.device)
    prompt = args.prompt
    max_tokens = args.n
    top_k = args.top_k
    model = Qwen25Model.from_pretrained()
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-0.5B-Instruct')
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([tokens], return_tensors="np")

    model_input_ids = dsc.from_numpy(model_inputs.input_ids.astype(np.int32), device='cpu')

    model.generate(model_input_ids, tokenizer, max_new_tokens=max_tokens, top_k=top_k)
