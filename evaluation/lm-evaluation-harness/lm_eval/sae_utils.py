import torch
import contextlib
import functools

from typing import List, Tuple, Callable
from torch import Tensor

from sae_lens import SAE


@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()


def get_intervention_hook(
    sae: SAE,
    feature_idx: int,
    max_activation: float = 1.0,
    strength: float = 1.0,
):
    def hook_fn(module, input, output):
        if torch.is_tensor(output):
            activations = output.clone()
        else:
            activations = output[0].clone()

        if sae.device != activations.device:
            sae.device = activations.device
            sae.to(sae.device)

        features = sae.encode(activations)
        reconstructed = sae.decode(features)
        error = activations.to(features.dtype) - reconstructed

        features[..., feature_idx] = max_activation * strength

        activations_hat = sae.decode(features) + error
        activations_hat = activations_hat.type_as(activations)

        if torch.is_tensor(output):
            return activations_hat
        else:
            return (activations_hat,) + output[1:] if len(output) > 1 else (activations_hat,)

    return hook_fn


def get_clamp_hook(
    direction: Tensor,
    max_activation: float = 1.0,
    strength: float = 1.0
):
    def hook_fn(module, input, output):
        nonlocal direction
        if torch.is_tensor(output):
            activations = output.clone()
        else:
            activations = output[0].clone()
        
        direction = direction / torch.norm(direction)
        direction = direction.type_as(activations)
        proj_magnitude = torch.sum(activations * direction, dim=-1, keepdim=True)
        orthogonal_component = activations - proj_magnitude * direction

        clamped = orthogonal_component + direction * max_activation * strength

        if torch.is_tensor(output):
            return clamped
        else:
            return (clamped,) + output[1:] if len(output) > 1 else (clamped,)

    return hook_fn
