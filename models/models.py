import contextlib
import logging
import re
from typing import Dict, List, Optional, Tuple

import torch
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
from torch.amp import autocast
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerFast

from utils.hpc import build_hf_load_args, cleanup_torch, configure_hpc_runtime, model_device, should_use_cuda_autocast
from utils.text import find_target_span  # noqa: F401  (re-exported; defined torch-free in utils/text.py)


logger = logging.getLogger(__name__)
configure_hpc_runtime()




_DECODER_ONLY_MODEL_TYPES = {
    'gpt2', 'gpt_neo', 'gpt_neox', 'gptj', 'bloom', 'opt',
    'llama', 'mistral', 'qwen2', 'olmo', 'falcon', 'gemma',
    'phi', 'stablelm', 'mpt', 'rwkv', 'codegen', 'xglm',
}


def is_decoder_only(model) -> bool:
    """Return True if model is a causal/decoder-only architecture."""
    cfg = model.config
    if getattr(cfg, 'is_decoder', False):
        return True
    archs = getattr(cfg, 'architectures', None) or []
    if any('causal' in a.lower() or 'generative' in a.lower() for a in archs):
        return True
    return getattr(cfg, 'model_type', '').lower() in _DECODER_ONLY_MODEL_TYPES


def _resolve_source_and_args(model_name: str, model_type: str) -> Tuple[str, Dict[str, object]]:
    runtime = build_hf_load_args(model_name, model_type=model_type)
    source = runtime["source"]
    load_args = dict(runtime["load_args"])
    return source, load_args


def _load_auto_model(source: str, load_args: Dict[str, object], requested_name: str):
    try:
        return AutoModel.from_pretrained(source, **load_args)
    except (RepositoryNotFoundError, HfHubHTTPError) as exc:
        raise ValueError(f"Model {requested_name} not found or inaccessible: {exc}") from exc
    except Exception as exc:
        if "custom code" in str(exc).lower() or "trust_remote_code" in str(exc).lower():
            retry_args = dict(load_args)
            retry_args["trust_remote_code"] = True
            logger.warning(
                "Custom code required for %s; retrying with trust_remote_code=True",
                requested_name,
            )
            return AutoModel.from_pretrained(source, **retry_args)
        raise


def _load_auto_tokenizer(source: str, load_args: Dict[str, object], requested_name: str):
    tokenizer_args = {
        key: value
        for key, value in load_args.items()
        if key in {"local_files_only", "token", "use_auth_token", "trust_remote_code"}
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(source, **tokenizer_args)
    except Exception as exc:
        if "custom code" in str(exc).lower() or "trust_remote_code" in str(exc).lower():
            tokenizer_args["trust_remote_code"] = True
            logger.warning(
                "Custom tokenizer code required for %s; retrying with trust_remote_code=True",
                requested_name,
            )
            tokenizer = AutoTokenizer.from_pretrained(source, **tokenizer_args)
        else:
            raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_tokenizer(model_name: str, model_type: str = "default"):
    source, load_args = _resolve_source_and_args(model_name, model_type)
    return _load_auto_tokenizer(source, load_args, model_name)


def load_model_and_tokenizer(model_name: str, model_type: str = "default"):
    source, load_args = _resolve_source_and_args(model_name, model_type)
    model = _load_auto_model(source, load_args, model_name)
    model.eval()
    tokenizer = _load_auto_tokenizer(source, load_args, model_name)
    return model, tokenizer


def _get_inference_context(device: torch.device):
    base = torch.inference_mode()
    if should_use_cuda_autocast() and device.type == "cuda":
        return contextlib.ExitStack().__enter__()
    return base


@contextlib.contextmanager
def _inference_context(device: torch.device):
    with torch.inference_mode():
        if should_use_cuda_autocast() and device.type == "cuda":
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                yield
        else:
            yield


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def get_target_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    texts: List[str],
    targets: List[str],
    batch_size: int = 8,
    layer_indices: List[int] = None,
    pooling: str = "target",
) -> Dict[int, torch.Tensor]:
    """
    Extract per-sentence hidden-state vectors across all layers.

    pooling='target'     — mean-pool the subword tokens that cover the target word.
                           Correct for bidirectional encoders that see full context.
    pooling='last_token' — take the last non-padding token's hidden state.
                           Use this for causal/decoder-only models: the target word
                           may appear before the disambiguating context, so the last
                           token position captures the full available left context.
    """
    if pooling not in ("target", "last_token"):
        raise ValueError(f"pooling must be 'target' or 'last_token', got {pooling!r}")

    device = model_device(model)
    num_hidden = model.config.num_hidden_layers
    if layer_indices is None:
        layer_indices = list(range(num_hidden + 1))

    all_acts = {layer: [] for layer in layer_indices}

    for start in range(0, len(texts), batch_size):
        batch_texts   = texts[start:start + batch_size]
        batch_targets = targets[start:start + batch_size]

        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        offset_mappings = encoding.pop("offset_mapping")
        attention_mask  = encoding["attention_mask"]
        model_inputs    = _move_batch_to_device(dict(encoding), device)

        with _inference_context(device):
            outputs = model(**model_inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        if pooling == "last_token":
            last_positions = (attention_mask.sum(dim=1) - 1).tolist()
        else:
            batch_masks: List[torch.Tensor] = []
            for offsets, target, text in zip(offset_mappings, batch_targets, batch_texts):
                offset_list = offsets.tolist()
                span = find_target_span(text, target)
                if span is None:
                    batch_masks.append(torch.zeros(len(offset_list), dtype=torch.bool))
                    continue
                target_start, target_end = span
                mask = [not (end <= target_start or begin >= target_end) for begin, end in offset_list]
                batch_masks.append(torch.tensor(mask, dtype=torch.bool))

        for layer in layer_indices:
            hidden = hidden_states[layer].detach().to(torch.float32).cpu()
            for sample_idx in range(hidden.size(0)):
                if pooling == "last_token":
                    pos = int(last_positions[sample_idx])
                    vec = hidden[sample_idx, pos : pos + 1, :]
                else:
                    mask = batch_masks[sample_idx]
                    if mask.any():
                        vec = hidden[sample_idx][mask].mean(dim=0, keepdim=True)
                    else:
                        vec = torch.zeros((1, hidden.size(-1)), dtype=torch.float32)
                all_acts[layer].append(vec)

        cleanup_torch()

    return {layer: torch.cat(all_acts[layer], dim=0) for layer in layer_indices}


def get_dual_position_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    texts: List[str],
    targets: List[str],
    batch_size: int = 8,
    layer_indices: List[int] = None,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Single forward pass that returns activations at two positions per sentence:

    target_acts  — mean-pooled hidden states at the target-word token(s).
                   For decoders this reflects only left context (causal constraint).
    final_acts   — hidden state at the last non-special, non-padding token.
                   For decoders this aggregates the full left context up to EOS.

    Returns (target_acts, final_acts) where each maps layer_idx → Tensor[N, D].

    H4 uses this to compare within-position sense decodability at the homonym
    and final positions. That is a token-position comparison from one forward
    pass, not evidence of incremental recovery or backtracking.

    Note: "final position" must skip trailing special tokens (e.g. [SEP] for
    BERT-family encoders). Those are structural markers whose hidden state does
    not vary meaningfully with sentence content, so naively taking the last
    non-padding position silently measures [SEP]'s representation instead of
    the sentence's actual final content token.
    """
    device = model_device(model)
    num_hidden = model.config.num_hidden_layers
    if layer_indices is None:
        layer_indices = list(range(num_hidden + 1))

    target_acts: Dict[int, List[torch.Tensor]] = {l: [] for l in layer_indices}
    final_acts:  Dict[int, List[torch.Tensor]] = {l: [] for l in layer_indices}

    for start in range(0, len(texts), batch_size):
        batch_texts   = texts[start:start + batch_size]
        batch_targets = targets[start:start + batch_size]

        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            add_special_tokens=True,
        )
        offset_mappings = encoding.pop("offset_mapping")
        special_tokens_mask = encoding.pop("special_tokens_mask")
        attention_mask  = encoding["attention_mask"]
        model_inputs    = _move_batch_to_device(dict(encoding), device)

        with _inference_context(device):
            outputs = model(**model_inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        # Build target-word masks (word-boundary safe)
        batch_masks: List[torch.Tensor] = []
        for offsets, target, text in zip(offset_mappings, batch_targets, batch_texts):
            offset_list = offsets.tolist()
            span = find_target_span(text, target)
            if span is None:
                batch_masks.append(torch.zeros(len(offset_list), dtype=torch.bool))
                continue
            target_start, target_end = span
            mask = [not (e <= target_start or b >= target_end) for b, e in offset_list]
            batch_masks.append(torch.tensor(mask, dtype=torch.bool))

        # Last non-special, non-padding token position per sample (skips trailing
        # [SEP]/[CLS]-style markers so "final position" is a real content token).
        content_mask = (attention_mask.bool()) & (special_tokens_mask == 0)
        row_lengths = attention_mask.sum(dim=1)
        last_positions = []
        for i, row in enumerate(content_mask):
            nonzero = row.nonzero(as_tuple=True)[0]
            if len(nonzero) > 0:
                last_positions.append(int(nonzero[-1]))
            else:
                last_positions.append(int(row_lengths[i]) - 1)

        for layer in layer_indices:
            hidden = hidden_states[layer].detach().to(torch.float32).cpu()
            for i, (tmask, last_pos) in enumerate(zip(batch_masks, last_positions)):
                # Target position
                if tmask.any():
                    tv = hidden[i][tmask].mean(dim=0, keepdim=True)
                else:
                    tv = torch.zeros((1, hidden.size(-1)), dtype=torch.float32)
                target_acts[layer].append(tv)

                # Final position
                fv = hidden[i, int(last_pos) : int(last_pos) + 1, :]
                final_acts[layer].append(fv)

        cleanup_torch()

    return (
        {l: torch.cat(target_acts[l], dim=0) for l in layer_indices},
        {l: torch.cat(final_acts[l],  dim=0) for l in layer_indices},
    )


def get_homonym_and_resolution_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    texts: List[str],
    homonym_targets: List[str],
    resolution_targets: List[str],
    batch_size: int = 8,
    layer_indices: List[int] = None,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Single forward pass returning activations at two content-word positions
    per sentence, both located via word-boundary-safe string search rather
    than sequence position:

    homonym_acts    — mean-pooled hidden states at the homonym token(s).
    resolution_acts — mean-pooled hidden states at the annotated
                      disambiguating word for that sentence (e.g. "river"
                      for a garden-path "bank" sentence), wherever in the
                      sentence it actually falls.

    Unlike get_dual_position_activations's "final position" (the last
    non-special token, whatever word that happens to be), this targets the
    specific word that carries the sense resolution. A garden-path sentence
    that happens to end on a neutral or even primed-sense-associated word
    (e.g. "...flew over the scaffolding") still gets scored at the word that
    actually resolves the sense ("wings"), not at whatever token is last.

    This helper is retained for cross-token diagnostics only. Comparing the
    homonym to a different resolution word changes token identity and cannot
    demonstrate revision. The corrected H5 reruns prefixes and reads the same
    appended sentinel via get_dual_position_activations.
    """
    device = model_device(model)
    num_hidden = model.config.num_hidden_layers
    if layer_indices is None:
        layer_indices = list(range(num_hidden + 1))

    homonym_acts:    Dict[int, List[torch.Tensor]] = {l: [] for l in layer_indices}
    resolution_acts: Dict[int, List[torch.Tensor]] = {l: [] for l in layer_indices}

    for start in range(0, len(texts), batch_size):
        batch_texts      = texts[start:start + batch_size]
        batch_homonyms   = homonym_targets[start:start + batch_size]
        batch_resolvers  = resolution_targets[start:start + batch_size]

        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        offset_mappings = encoding.pop("offset_mapping")
        model_inputs    = _move_batch_to_device(dict(encoding), device)

        with _inference_context(device):
            outputs = model(**model_inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        def _span_mask(offsets, target, text):
            offset_list = offsets.tolist()
            span = find_target_span(text, target)
            if span is None:
                return torch.zeros(len(offset_list), dtype=torch.bool)
            target_start, target_end = span
            mask = [not (e <= target_start or b >= target_end) for b, e in offset_list]
            return torch.tensor(mask, dtype=torch.bool)

        homonym_masks    = [_span_mask(o, t, txt) for o, t, txt in zip(offset_mappings, batch_homonyms, batch_texts)]
        resolution_masks = [_span_mask(o, t, txt) for o, t, txt in zip(offset_mappings, batch_resolvers, batch_texts)]

        for layer in layer_indices:
            hidden = hidden_states[layer].detach().to(torch.float32).cpu()
            for i, (hmask, rmask) in enumerate(zip(homonym_masks, resolution_masks)):
                if hmask.any():
                    hv = hidden[i][hmask].mean(dim=0, keepdim=True)
                else:
                    hv = torch.zeros((1, hidden.size(-1)), dtype=torch.float32)
                homonym_acts[layer].append(hv)

                if rmask.any():
                    rv = hidden[i][rmask].mean(dim=0, keepdim=True)
                else:
                    rv = torch.zeros((1, hidden.size(-1)), dtype=torch.float32)
                resolution_acts[layer].append(rv)

        cleanup_torch()

    return (
        {l: torch.cat(homonym_acts[l],    dim=0) for l in layer_indices},
        {l: torch.cat(resolution_acts[l], dim=0) for l in layer_indices},
    )
