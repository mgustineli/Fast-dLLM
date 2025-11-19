import torch
import types
from typing import Dict, List, Optional, Union


def _shape_of(x):
    """Return tensor shape as tuple if x is a torch.Tensor, else None."""
    try:
        if isinstance(x, torch.Tensor):
            return tuple(x.shape)
    except Exception:
        pass
    return None


def _get_seq_len_from_args(args, kwargs):
    """
    Try to extract the sequence length (T) from the first tensor argument
    (usually hidden_states: [B, T, D]).
    """
    if args and _shape_of(args[0]) is not None and len(args[0].shape) >= 2:
        try:
            return int(args[0].shape[1])
        except Exception:
            pass
    hs = kwargs.get("hidden_states", None)
    if hs is not None and _shape_of(hs) is not None and len(hs.shape) >= 2:
        try:
            return int(hs.shape[1])
        except Exception:
            pass
    return None


class LayerReuseController:
    """
    This controller wraps a subset of decoder layers to reuse their outputs
    for (reuse_k - 1) forward passes and recompute them every kth pass.
    """

    def __init__(self, model_with_layers, subset: Optional[Union[str, List[int]]], reuse_k: int):
        """
        Args:
            model_with_layers: model exposing .layers or .model.layers
            subset: which layers to reuse ("first", "middle", "last", list of indices, or None)
            reuse_k: number of steps between recomputations
        """
        # detect where the layer stack lives
        if hasattr(model_with_layers, "model") and hasattr(
            model_with_layers.model, "layers"
        ):
            self.layers = model_with_layers.model.layers
        else:
            self.layers = model_with_layers.layers

        self.reuse_k = max(int(reuse_k), 1)
        self.enabled = False
        self.counter = 0
        self.cache: Dict[int, object] = {}  # stores cached layer outputs
        self.orig_forwards: Dict[int, object] = {}  # keeps original forward methods

        n = len(self.layers)
        subset_size = min(12, n)

        # select subset of layers to apply reuse on
        if subset is None:
            # --- FIX 1: Default to ALL layers if None is passed ---
            # This ensures that if you just want to test 'k', it applies to the whole model.
            self.reuse_layers = list(range(n))
        elif subset == "first":
            self.reuse_layers = list(range(0, subset_size))
        elif subset == "middle":
            start = max(0, n // 2 - subset_size // 2)
            self.reuse_layers = list(range(start, start + subset_size))
        elif subset == "last":
            self.reuse_layers = list(range(n - subset_size, n))
        else:
            self.reuse_layers = list(subset)

    def _make_wrapper(self, layer_idx: int, orig_forward):
        """
        Creates a wrapped version of the layer's forward method.
        """
        is_bound = (
            hasattr(orig_forward, "__self__") and orig_forward.__self__ is not None
        )

        def wrapped(self_layer, *args, **kwargs):
            # extract sequence length (T)
            T_cur = _get_seq_len_from_args(args, kwargs)

            # --- FIX 2: Refine Stateful Logic ---
            # We ONLY want to block reuse if we are modifying the GLOBAL past_key_values.
            # We do NOT want to block reuse just because use_block_cache is True, 
            # because that flag is active during the exact loop where we want reuse to happen.
            
            has_global_past = kwargs.get("past_key_value", None) is not None
            
            # Note: We ignore 'use_block_cache' and 'block_past_key_values' here.
            # The generation loop handles the logic of when to flush the block cache.
            # This controller's job is strictly layer-output reuse.

            # We also check if 'update_past_key_values' is explicitly True (usually prefill)
            updating_past = kwargs.get("update_past_key_values", False)

            # If we are strictly updating global state, we shouldn't reuse.
            stateful = has_global_past and updating_past

            # check if reuse is valid for this layer
            can_reuse = (
                self.enabled
                and (layer_idx in self.reuse_layers)
                and (self.reuse_k > 1)
                and (self.counter % self.reuse_k != 0)
                and not stateful
            )

            # guard against sequence length changes (sanity check)
            meta = self.cache.get(("meta", layer_idx))
            last_T = meta.get("T") if isinstance(meta, dict) else None
            
            # If T_cur is None (couldn't detect), we assume it's unsafe to reuse unless we are sure
            same_len = (T_cur is not None) and (last_T is not None) and (T_cur == last_T)

            # use cached output if safe and available
            if can_reuse and same_len and (layer_idx in self.cache):
                return self.cache[layer_idx]

            # otherwise, compute output normally
            out = (
                orig_forward(*args, **kwargs)
                if is_bound
                else orig_forward(self_layer, *args, **kwargs)
            )

            # store output:
            # We store it if enabled, inside the subset, and we are NOT in a stateful global update.
            if (
                self.enabled
                and (layer_idx in self.reuse_layers)
                and (not stateful)
            ):
                # We allow caching even if T changes, we just overwrite the old one.
                self.cache[layer_idx] = out
                self.cache[("meta", layer_idx)] = {"T": T_cur}

            return out

        # preserve original metadata
        wrapped.__orig_forward__ = orig_forward
        wrapped.__controller__ = self
        return wrapped

    def enable_reuse(self):
        """Enable layer reuse by patching selected layers' forward methods."""
        if self.enabled:
            return
        self.enabled = True
        self.counter = 0
        self.cache.clear()

        for idx, layer in enumerate(self.layers):
            if idx in self.reuse_layers:
                self.orig_forwards[idx] = layer.forward
                layer.forward = types.MethodType(
                    self._make_wrapper(idx, layer.forward), layer
                )

    def disable_reuse(self):
        """Restore original forward methods and clear the reuse cache."""
        if not self.enabled:
            return
        for idx, layer in enumerate(self.layers):
            if idx in self.reuse_layers and idx in self.orig_forwards:
                layer.forward = self.orig_forwards[idx]
        self.orig_forwards.clear()
        self.cache.clear()
        self.enabled = False
        self.counter = 0

    def step(self):
        """
        Should be called once after each decoding step.
        """
        if not self.enabled:
            return
        self.counter += 1
        # We do NOT clear the cache here anymore.
        # Why? Because we might want to reuse the cache for k steps.
        # If we clear it, we force a recompute next step regardless of 'k'.
        # We only need to clear if the sequence length changes drastically or context shifts,
        # but the _make_wrapper logic handles overwrites.
        
        # However, if you want to enforce a hard flush every k steps to be safe:
        if self.counter % self.reuse_k == 0:
            self.cache.clear()
