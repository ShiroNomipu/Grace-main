"""Backbone-agnostic accessors, added for the cross-backbone control runs.

The adapter is injected by hooking decoder layers of the frozen LLM, which
means three things have to be read off the loaded model rather than hard-coded
to Qwen2-VL-2B: which class loads it, where its decoder layers live, and how
wide its residual stream is. Everything else in the pipeline is already
backbone-agnostic, because prompts go through `tokenizer.apply_chat_template`
and the LoRA target modules use the q/k/v/o/gate/up/down naming that both
families share.

Defaults are unchanged for Qwen2-VL, so existing runs reproduce bit for bit.
"""
import torch.nn as nn
from transformers import AutoConfig

# Attribute chains tried in order. Qwen2-VL nests the decoder under
# `language_model`; Llama does not; PEFT wraps either in `base_model.model`.
_CHAINS = (
    "model.language_model.layers",
    "model.model.language_model.layers",
    "model.layers",
    "model.model.layers",
    "base_model.model.model.layers",
    "base_model.model.model.language_model.layers",
)


def _resolve(obj, chain):
    for attr in chain.split("."):
        if not hasattr(obj, attr):
            return None
        obj = getattr(obj, attr)
    return obj


def text_config(cfg):
    """Qwen2-VL keeps the LLM dims under `text_config`; Llama keeps them flat."""
    return getattr(cfg, "text_config", cfg)


def decoder_layers(model):
    """Return the ModuleList of decoder layers of the language model.

    The candidate chains resolve differently across transformers versions and
    through PEFT's attribute forwarding, so every hit is checked against the
    depth declared in the config: hooking the wrong ModuleList would change
    where the injection happens without raising anything.
    """
    want = num_layers(model)
    for chain in _CHAINS:
        got = _resolve(model, chain)
        if isinstance(got, nn.ModuleList) and len(got) == want:
            return got
    # Fall back to the longest ModuleList of identically-typed blocks, so an
    # unfamiliar architecture fails with a clear message rather than silently
    # hooking the wrong module.
    for _name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == want:
            types = {type(m).__name__ for m in mod}
            if len(types) == 1 and "Layer" in next(iter(types)):
                return mod
    raise RuntimeError(
        f"could not locate the {want} decoder layers of "
        f"{type(model).__name__}; add its attribute chain to backbone._CHAINS")


def hidden_size(model):
    return text_config(model.config).hidden_size


def num_layers(model):
    return text_config(model.config).num_hidden_layers


def describe(model_path):
    """Read depth/width from a checkpoint directory without loading weights."""
    tc = text_config(AutoConfig.from_pretrained(model_path))
    return {
        "model_type": tc.model_type,
        "num_hidden_layers": tc.num_hidden_layers,
        "hidden_size": tc.hidden_size,
        "num_attention_heads": tc.num_attention_heads,
    }


def insert_layers_for(n_blocks, n_layers):
    """Insertion depths for a backbone with `n_layers` decoder layers.

    The main line uses [9, 18] of 28 layers. On a backbone of the same depth
    those indices are kept exactly, so a cross-backbone run changes the family
    and width without also changing where the injection happens. On a different
    depth they are rescaled to the same relative positions.
    """
    ref_total = 28
    ref = {2: [9, 18], 4: [4, 9, 14, 18], 6: [3, 7, 11, 15, 19, 23]}
    if n_blocks in ref:
        base = ref[n_blocks]
        if n_layers == ref_total:
            return list(base)
        return [max(0, min(n_layers - 1, round(i * n_layers / ref_total))) for i in base]
    step = n_layers // (n_blocks + 1)
    return [step * (i + 1) for i in range(n_blocks)]
