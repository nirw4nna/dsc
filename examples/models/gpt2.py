#  Copyright (c) 2024-2025, Christian Gilli <christian.gilli@dspcraft.com>
#  All rights reserved.
#
#  This code is licensed under the terms of the 3-clause BSD license
#  (https://opensource.org/license/bsd-3-clause).


import dsc
import dsc.nn as nn
from dataclasses import dataclass
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from time import perf_counter


@dataclass
class GPT2Hparams:
   # default hyperparameters for GPT-2 small
   n_layers: int = 12
   n_heads: int = 12
   emb_size: int = 768
   block_size: int = 1024
   vocab_size: int = 50257


class MultiHeadAttention(nn.Module):
    def __init__(self, hparams: GPT2Hparams, use_cache: bool = True):
        super().__init__()
        self.block_size = hparams.block_size
        self.emb_size = hparams.emb_size
        self.n_heads = hparams.n_heads
        # Stacked attention, contains the projections of both Q, K and V
        self.c_attn = nn.Linear(self.emb_size, 3 * self.emb_size)
        self.c_proj = nn.Linear(self.emb_size, self.emb_size)
        # Causal mask
        self.tril = dsc.tril(dsc.ones((self.block_size, self.block_size)))
        
        # KV cache
        self.use_cache = use_cache
        self.cache_k = None
        self.cache_v = None

    def forward(self, x: dsc.Tensor) -> dsc.Tensor:
        B, T, C = x.shape # (block size, context size, emb size)
        attn = self.c_attn(x)

        q, k, v = attn.split(self.emb_size, axis=2) # (B, T, C)
        q = q.reshape(B, T, self.n_heads, self.emb_size // self.n_heads).transpose((0, 2, 1, 3)) # (B, nh, T, hs) given EMB_SIZE is a multiple of N_HEADS
        k = k.reshape(B, T, self.n_heads, self.emb_size // self.n_heads).transpose((0, 2, 1, 3)) # (B, nh, T, hs)
        v = v.reshape(B, T, self.n_heads, self.emb_size // self.n_heads).transpose((0, 2, 1, 3)) # (B, nh, T, hs)

        if self.use_cache:
            if self.cache_k is not None:
                k = dsc.concat([self.cache_k, k], axis=2)

            if self.cache_v is not None:
                v = dsc.concat([self.cache_v, v], axis=2)

            self.cache_k = k
            self.cache_v = v

        seq_len = k.size(2)
        k_t = k.transpose((0, 1, 3, 2))

        # Self Attention (B, nh, T, hs) @ (B, nh, hs, T) = (B, nh, T, T)
        q_k = q @ k_t
        attention = q_k * q.size(-1) ** -0.5

        if not self.use_cache or seq_len == T:
            # Masking is needed when we are not using the cache or when using the cache and we are processing the prompt
            attention = attention.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        attention = nn.softmax(attention, axis=-1)
        out = attention @ v # (B, nh, T, T) @ (B, nh, T, hs) = (B, nh, T, hs)
        out = out.transpose((0, 2, 1, 3)).reshape(B, T, C)

        return self.c_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, hparams: GPT2Hparams, use_cache: bool = True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hparams.emb_size)
        self.attn = MultiHeadAttention(hparams, use_cache)
        self.ln_2 = nn.LayerNorm(hparams.emb_size)
        self.mlp = nn.ModuleDict(dict(
            c_fc = nn.Linear(hparams.emb_size, hparams.emb_size * 4),
            c_proj = nn.Linear(hparams.emb_size * 4, hparams.emb_size),
        ))

    def forward(self, x):
        m = self.mlp
        x = x + self.attn(self.ln_1(x))
        return x + m.c_proj(nn.gelu(m.c_fc(self.ln_2(x))))


class GPT2(nn.Module):
    def __init__(self, hparams: GPT2Hparams, use_cache: bool = True):
        super().__init__()
        self.hparams = hparams
        self.transformer = nn.ModuleDict(dict(
            wpe = nn.Embedding(hparams.block_size, hparams.emb_size),
            wte = nn.Embedding(hparams.vocab_size, hparams.emb_size),
            h = nn.ModuleList([TransformerBlock(hparams, use_cache) for _ in range(hparams.n_layers)]),
            ln_f = nn.LayerNorm(hparams.emb_size)
        ))
        self.lm_head = nn.Linear(hparams.emb_size, hparams.vocab_size, bias=False)
        self.use_cache = use_cache
        self.kv_pos = 0

        n_params = sum([p.ne for p in self.transformer.parameters()])
        n_params += sum([p.ne for p in self.lm_head.parameters()])
        print(f'Model has {round(n_params / 1e6)}M parameters')
    
    @staticmethod
    def from_hf(hparams: GPT2Hparams = GPT2Hparams(), use_cache: bool = True) -> 'GPT2':
        hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
        # GPT2 uses Conv1D instead of a Linear layer which means we have to transpose the weights
        to_transpose = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        my_model = GPT2(hparams, use_cache)
        my_model.from_torch(hf_model, on_hook=[
            (to_transpose, lambda x: x.t().contiguous())
        ])
        del hf_model
        return my_model

    def forward(self, idx: dsc.Tensor) -> dsc.Tensor:
        B, T = idx.shape
        tok_emb = self.transformer.wte(idx)
        if self.use_cache:
            pos_emb = self.transformer.wpe(dsc.arange(T) + self.kv_pos)
        else:
            pos_emb = self.transformer.wpe(dsc.arange(T))
    
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        self.kv_pos += T
        return logits

    def generate(self, idx: dsc.Tensor, max_new_tokens: int, temp: float = 1, sample: bool = True) -> dsc.Tensor:
        assert max_new_tokens < self.hparams.block_size
        # Include the input in the response
        generated = idx
        # The first time process the entire prompt then only the last token
        idx_next = idx
        for _ in range(max_new_tokens):
            if self.use_cache:
                logits = self(idx_next)
            else:
                logits = self(generated)
            # Apply temperature to the last row of each bach
            logits = logits[:, -1, :] * (1 / temp)
            v, _ = dsc.topk(logits, 10)
            # NOTE: the point here is that I want v[:, -1] to be broadcast to the entire logits tensor
            # in DSC this is the case if v[:, -1] is a scalar (ie. 1D with 1 element)
            logits = logits.masked_fill(logits < v[:, -1], -float('Inf'))
            probs = nn.softmax(logits, axis=-1)
            if sample:
                idx_next = dsc.multinomial(probs, num_samples=1)
            else:
                _, idx_next = dsc.topk(probs, k=1, axis=-1)
            
            generated = dsc.concat([generated, idx_next], axis=1)

        return generated


if __name__ == '__main__':
    model = GPT2.from_hf(use_cache=True)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    prompt = 'Javascript is a programming language designed'
    print(f'[In]: {prompt}')
    
    MAX_TOKENS = 10

    idx = tokenizer.encode(prompt)
    start = perf_counter()

    response_tokens = model.generate(dsc.tensor(idx, dtype=dsc.Dtype.I32).reshape(1, -1), max_new_tokens=MAX_TOKENS)
    delay = perf_counter() - start

    print(f'[Out]: {tokenizer.decode(response_tokens[0].numpy().tolist())}')
    print(f'Took {round(delay, 2)}s to generate {MAX_TOKENS} tokens ({round(MAX_TOKENS / delay, 2)} tok/s)')
