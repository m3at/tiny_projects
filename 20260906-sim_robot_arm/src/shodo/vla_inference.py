"""Exact-input inference caches for a frozen, pinned SmolVLA policy.

Camera keys identify immutable acquired frames and must change for every new
acquisition. Clear caches on episode reset or model/device/preprocessing changes.
Processed VLM prefix keys/values are never cached: they contain measured state.
"""

from contextlib import contextmanager

import torch


class PromptCache:
    """Cache tokenization only, keyed by the complete task and tokenizer options."""

    def __init__(self, tokenizer, *, trim_padding=False):
        if type(trim_padding) is not bool:
            raise ValueError("trim_padding must be boolean")
        self.tokenizer = tokenizer
        self.trim_padding = trim_padding
        self.clear()

    def clear(self):
        self.key = None
        self.tokens = None

    def __call__(self, tasks, **kwargs):
        key = (tuple(tasks), tuple(sorted(kwargs.items())), self.trim_padding)
        if key != self.key:
            self.tokens = self.tokenizer(tasks, **kwargs)
            if self.trim_padding:
                mask = self.tokens["attention_mask"]
                ids = self.tokens["input_ids"]
                if mask.ndim != 2 or ids.shape != mask.shape or not mask.shape[1]:
                    raise ValueError("Token IDs and masks must be matching nonempty matrices")
                # Keep internal/left padding and every column used by any batch item.
                # Retain one column for the degenerate all-masked batch.
                valid = mask.bool().any(dim=0).nonzero()
                width = int(valid[-1, 0]) + 1 if len(valid) else 1
                self.tokens = {
                    **self.tokens,
                    "input_ids": ids[:, :width],
                    "attention_mask": mask[:, :width],
                }
            self.key = key
        return self.tokens


class FrameCache:
    """Cache one vision embedding, never state projection or the VLM prefix.

    The caller sets frame to an immutable acquisition identifier before inference.
    Use only for frozen-model evaluation under torch.no_grad/inference_mode; cached
    output must not be mutated. This cache intentionally performs no GPU equality
    test, host transfer, or hash of image contents in the inference hot path.
    """

    def __init__(self, encode):
        self.encode = encode
        self.clear()

    def clear(self):
        self.frame = None
        self.cached_frame = None
        self.value = None
        self.hits = 0
        self.misses = 0

    def __call__(self, image):
        owner = getattr(self.encode, "__self__", None)
        if torch.is_grad_enabled() or getattr(owner, "training", False):
            raise RuntimeError(
                "Vision caching requires frozen evaluation under no_grad/inference_mode"
            )
        if self.frame is None:
            raise ValueError("Set the acquisition index before image encoding")
        if self.cached_frame != self.frame:
            self.value = self.encode(image)
            self.cached_frame = self.frame
            self.misses += 1
        else:
            self.hits += 1
        return self.value


@contextmanager
def cached_vision(model):
    """Temporarily cache a frozen, merged SmolVLA policy's image encoder.

    A scope owns one policy instance; this mutating hook is not thread-safe. It
    restores the original method on exit without changing upstream source/weights.
    """
    vision = model.model.vlm_with_expert
    original = vision.embed_image
    cache = FrameCache(original)
    vision.embed_image = cache
    try:
        yield cache
    finally:
        vision.embed_image = original


class StaticPrefixCache:
    """Cache only image/language input embeddings; always project current state.

    This targets the pinned single-camera, unpadded SmolVLA input layout. Set
    frame and task before each call; those identifiers must cover immutable raw
    acquisitions and tokenization settings. Clear on reset or weight changes.
    Full VLM processing, including its state-dependent K/V, runs on every call.
    """

    def __init__(self, model):
        self.model = model
        self.encode = model.embed_prefix
        self.clear()

    def clear(self):
        self.frame = None
        self.task = None
        self.cached_key = None
        self.static = None
        self.pad_mask = None
        self.att_mask = None
        self.hits = 0
        self.misses = 0

    def __call__(self, images, img_masks, lang_tokens, lang_masks, state=None):
        model = self.model
        if torch.is_grad_enabled() or model.training:
            raise RuntimeError(
                "Static embedding caching requires frozen evaluation under no_grad/inference_mode"
            )
        if model.prefix_length != 0 or model.config.prefix_length != 0:
            raise ValueError(
                "Static embedding caching requires prefix_length=0 (no prefix padding)"
            )
        if self.frame is None or not isinstance(self.task, str) or not self.task:
            raise ValueError("Set immutable frame and task identifiers before prefix encoding")
        if (
            not isinstance(state, torch.Tensor)
            or state.ndim not in (2, 3)
            or (state.ndim == 3 and state.shape[1] != 1)
        ):
            raise ValueError("Static embedding caching requires exactly one current state token")
        if len(images) != 1 or len(img_masks) != 1:
            raise ValueError("Static embedding caching supports exactly one camera")
        if lang_tokens.ndim != 2 or lang_masks.shape != lang_tokens.shape:
            raise ValueError("Language tokens and masks must be matching matrices")
        signature = tuple(
            (value.shape, value.dtype, value.device)
            for value in (
                state,
                images[0],
                img_masks[0],
                lang_tokens,
                lang_masks,
                model.state_proj.weight,
            )
        )
        key = (self.frame, self.task, model.add_image_special_tokens, signature)
        if key != self.cached_key:
            prefix, pad_mask, att_mask = self.encode(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
            if (
                prefix.ndim != 3
                or prefix.shape[0] != state.shape[0]
                or prefix.shape[1] < 2
                or pad_mask.shape != prefix.shape[:2]
                or att_mask.shape != pad_mask.shape
                or pad_mask.dtype != torch.bool
                or att_mask.dtype != torch.bool
            ):
                raise ValueError("Unexpected pinned SmolVLA prefix layout")
            # prefix_length=0 and one input state token put current state last.
            # Clone retained values so callers cannot corrupt future cache hits.
            self.static = prefix[:, :-1].clone()
            self.pad_mask = pad_mask.clone()
            self.att_mask = att_mask.clone()
            self.cached_key = key
            self.misses += 1
            return prefix, pad_mask, att_mask
        state_embedding = model.state_proj(state)
        if state_embedding.ndim == 2:
            state_embedding = state_embedding[:, None, :]
        self.hits += 1
        return (
            torch.cat((self.static, state_embedding), dim=1),
            self.pad_mask.clone(),
            self.att_mask.clone(),
        )


@contextmanager
def cached_static_prefix(model):
    """Scope an inference-only static input cache to one frozen, merged policy."""
    flow = model.model
    original = flow.embed_prefix
    cache = StaticPrefixCache(flow)
    flow.embed_prefix = cache
    try:
        yield cache
    finally:
        flow.embed_prefix = original
