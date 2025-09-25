import os
import json
import fire
from dataclasses import dataclass
from typing import List, Tuple, Iterable
from jaxtyping import Int, Float, Bool

from collections import defaultdict
import re
import math
import torch
from torch import Tensor
from datasets import load_dataset

from tqdm import tqdm

from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE
from sae_lens.config import DTYPE_MAP as DTYPES
from sae_lens.sae import TopK
from sae_dashboard.feature_data_generator import FeatureMaskingContext


@dataclass
class SaeSelectionConfig:
    hook_point: str
    features: Iterable[int]
    minibatch_size_features: int = 256
    minibatch_size_tokens: int = 64
    device: str = "cpu"
    dtype: str = "float32"


def split_data(data, num_parts):
    """
    Split `data` into `num_parts` batches
    """
    k, m = divmod(len(data), num_parts)
    batches = [data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_parts)]
    return batches


class RollingMean:
    def __init__(
        self, 
        tokens_of_interest: List[List[Tensor]], 
        ignore_tokens: List[int] = None,
        expand_range: Tuple[int, int] = None
    ):
        self.tokens_of_interest = tokens_of_interest
        self.ignore_tokens = ignore_tokens if ignore_tokens is not None else []
        self.expand_range = expand_range if expand_range is not None else (0, 0)

        # single-level statistics
        self._means = None
        self._counts = [0] * len(self.tokens_of_interest)
        
        # whole positive statistics
        self._mean_pos = None
        self._count_pos = 0

        # whole negative statistics
        self._mean_neg = None
        self._count_neg = 0

    def _compute_single_mask(self, tokens: Int[Tensor, "batch seq"], ids_of_interest: Tensor):
        """Compute mask for a single token sequence with expansion."""
        seq_len = tokens.size(1)
        ids_len = len(ids_of_interest)
        
        mask = torch.zeros_like(tokens, dtype=torch.bool, device=tokens.device)
        if ids_len > seq_len:
            return mask

        ids_of_interest = ids_of_interest.view(1, 1, -1)
        windows = tokens.unfold(1, ids_len, 1)
        matches = (windows == ids_of_interest).all(dim=2)
        batch_indices, window_indices = torch.nonzero(matches, as_tuple=True)
        if len(batch_indices) == 0:
            return mask

        offsets = torch.arange(ids_len, device=tokens.device)
        spans = window_indices.unsqueeze(1) + offsets.unsqueeze(0)
        batch_expanded = batch_indices.unsqueeze(1).expand(-1, ids_len).reshape(-1)
        spans_flat = spans.reshape(-1)
        mask[batch_expanded, spans_flat] = True

        # Apply expand_range
        left, right = self.expand_range
        if left != 0 or right != 0:
            batch_indices, pos_indices = torch.nonzero(mask, as_tuple=True)
            if len(pos_indices) > 0:
                starts = torch.clamp(pos_indices - left, min=0)
                ends = torch.clamp(pos_indices + right, max=tokens.size(1) - 1)
                delta = torch.zeros(tokens.size(0), tokens.size(1) + 1, dtype=torch.int32, device=tokens.device)
                delta[batch_indices, starts] += 1
                delta[batch_indices, ends + 1] -= 1
                coverage = delta.cumsum(dim=1)
                coverage = coverage[:, :-1]  # Trim last column
                mask = coverage > 0

        return mask

    def _compute_update(
        self, tokens: Int[Tensor, "batch seq"], feature_acts: Float[Tensor, "batch seq n"],
        mask: Bool[Tensor, "batch seq"], acc_mean: Float[Tensor, "n"], acc_count: float
    ):
        tokens = tokens[mask]
        feature_acts = feature_acts[mask]

        if tokens.numel() == 0:
            return acc_mean, acc_count

        mean = feature_acts.mean(dim=0)
        count = feature_acts.size(0)

        upd_count = acc_count + count
        upd_mean = acc_mean + (count / upd_count) * (mean - acc_mean)

        return upd_mean, upd_count
    
    def update(self, tokens: Int[Tensor, "batch seq"], feature_acts: Float[Tensor, "batch seq n"]):
        assert tokens.ndim == 2 and feature_acts.ndim == 3, "tokens should be 2D, feature acts - 3D"
        assert tokens.size() == feature_acts.size()[:-1], "Batch and sequence dimensions must match"
        
        n = feature_acts.size(-1)

        if self._means is None:
            device = feature_acts.device
            dtype = feature_acts.dtype

            self._means = [
                torch.zeros(n, dtype=dtype, device=device)
                for _ in self.tokens_of_interest
            ]
            self._mean_pos = torch.zeros(n, dtype=dtype, device=device)
            self._mean_neg = torch.zeros(n, dtype=dtype, device=device)

            self.tokens_of_interest = [
                [seq.to(device) for seq in seq_group]
                for seq_group in self.tokens_of_interest
            ]
        
        if len(self.ignore_tokens) > 0:
            ignore_tensor = torch.tensor(self.ignore_tokens, dtype=torch.long, device=tokens.device)
            ignore_mask = torch.isin(tokens, ignore_tensor)
        else:
            ignore_mask = torch.zeros_like(tokens, dtype=torch.bool)
        
        mask_combined = torch.zeros_like(tokens, dtype=torch.bool)
        for i, seq_group in enumerate(self.tokens_of_interest):
            # Compute mask for one token
            group_mask = torch.zeros_like(tokens, dtype=torch.bool)
            for seq in seq_group:
                seq_mask = self._compute_single_mask(tokens, seq)
                group_mask |= seq_mask
            # Exclude ignored tokens
            group_mask = group_mask & (~ignore_mask)
            # Update rolling mean and count for the token
            self._means[i], self._counts[i] = self._compute_update(
                tokens, feature_acts, group_mask, self._means[i], self._counts[i]
            )
            # update mask
            mask_combined |= group_mask

        # Update 'positive' stats
        mask_pos = mask_combined & (~ignore_mask)
        self._mean_pos, self._count_pos = self._compute_update(
            tokens, feature_acts, mask_pos, self._mean_pos, self._count_pos
        )
        
        # Update 'negative' stats
        mask_neg = (~mask_combined) & (~ignore_mask)
        self._mean_neg, self._count_neg = self._compute_update(
            tokens, feature_acts, mask_neg, self._mean_neg, self._count_neg
        )

    def stats(self):
        """Returns tensors of shape [n_features, 2] & [n_features, n_reason_tokens]."""
        single_means = torch.stack(self._means, dim=1) if len(self._means) > 0 else torch.tensor([])
        
        means = torch.stack([self._mean_pos, self._mean_neg], dim=1)

        return means, single_means


class FeatureStatisticsGenerator:
    """Generator used to accumulate reasoning-related statistics for a batch of features. 
    
    Highly inspired by `sae_dashboard.FeatureDataGenerator`.
    """
    def __init__(
        self,
        cfg: SaeSelectionConfig,
        model: HookedTransformer,
        encoder: SAE,
        tokens: Int[Tensor, "batch seq"],
        reason_tokens: List[List[Tensor]],
        ignore_tokens: List[int] = None,
        expand_range: Tuple[int, int] = None
    ):
        self.cfg = cfg
        self.model = model
        self.encoder = encoder
        self.token_minibatches = self.batch_tokens(tokens)
        self.reason_tokens = reason_tokens
        self.ignore_tokens = ignore_tokens
        self.expand_range = expand_range
        self.hook_layer = self.get_layer(self.cfg.hook_point)

    @torch.inference_mode()
    def batch_tokens(
        self, tokens: Int[Tensor, "batch seq"]
    ) -> list[Int[Tensor, "batch seq"]]:
        # Get tokens into minibatches, for the fwd pass
        token_minibatches = (
            (tokens,)
            if self.cfg.minibatch_size_tokens is None
            else tokens.split(self.cfg.minibatch_size_tokens)
        )
        token_minibatches = [tok.to(self.cfg.device) for tok in token_minibatches]

        return token_minibatches

    def get_layer(self, hook_point: str):
        """Get the layer (so we can do the early stopping in our forward pass)"""
        layer_match = re.match(r"blocks\.(\d+)\.", hook_point)
        assert (
            layer_match
        ), f"Error: expecting hook_point to be 'blocks.{{layer}}.{{...}}', but got {hook_point!r}"
        return int(layer_match.group(1))

    @torch.inference_mode()
    def get_feature_data(
        self,
        feature_indices: list[int],
    ):
        # Create objects to store the data for computing rolling stats
        feature_means = RollingMean(self.reason_tokens, self.ignore_tokens, self.expand_range)

        for i, minibatch in tqdm(
            enumerate(self.token_minibatches), desc="Statistics aggregation", 
            total=len(self.token_minibatches), leave=False
        ):
            minibatch.to(self.cfg.device)
            model_acts = self.get_model_acts(minibatch).to(self.encoder.device)

            # For TopK, compute all activations first, then select features
            if isinstance(self.encoder.activation_fn, TopK):
                # Get all features' activations
                all_features_acts = self.encoder.encode(model_acts)
                # Then select only the features we're interested in
                feature_acts = all_features_acts[:, :, feature_indices].to(
                    DTYPES[self.cfg.dtype]
                )
            else:
                # For other activation functions, use the masking context
                with FeatureMaskingContext(self.encoder, feature_indices):
                    feature_acts = self.encoder.encode(model_acts).to(
                        DTYPES[self.cfg.dtype]
                    )

            feature_means.update(minibatch, feature_acts)
            
        agg_means, agg_single_means = feature_means.stats()

        return agg_means, agg_single_means

    @torch.inference_mode()
    def get_model_acts(
        self, tokens: Int[Tensor, "batch seq"]
    ):
        def hook_fn_store_act(activation: Tensor, hook: HookPoint):
            hook.ctx["activation"] = activation

        hooks = [(self.cfg.hook_point, hook_fn_store_act)]

        self.model.run_with_hooks(
            tokens, stop_at_layer=self.hook_layer + 1, 
            fwd_hooks=hooks, return_type=None
        )

        activation = self.model.hook_dict[self.cfg.hook_point].ctx.pop(
            "activation"
        )

        return activation


class SaeSelectionRunner:
    """Runner used to collect reasoning-related statistics.
    
    Highly inspired by `sae_dashboard.SaeVisRunner`.
    """
    def __init__(self, cfg: SaeSelectionConfig):
        self.cfg = cfg

    @torch.inference_mode()
    def run(
        self, 
        encoder: SAE,
        model: HookedTransformer, 
        tokens: Int[Tensor, "batch seq"],
        reason_tokens: List[List[Tensor]],
        ignore_tokens: List[int] = None,
        expand_range: Tuple[int, int] = None,
        alpha: float = 1.0,
        epsilon: float = 1e-12
    ):
        encoder.fold_W_dec_norm()

        features_list = self.handle_features(self.cfg.features, encoder)
        feature_batches = self.get_feature_batches(features_list)

        feature_statistics_generator = FeatureStatisticsGenerator(
            self.cfg, model, encoder, tokens, reason_tokens, ignore_tokens, expand_range
        )

        all_feature_means, all_feature_single_means = [], []
        for features in tqdm(feature_batches, total=len(feature_batches), desc="Feature Selection"):
            feature_means, feature_single_means = feature_statistics_generator.get_feature_data(features)

            all_feature_means.append(feature_means)
            all_feature_single_means.append(feature_single_means)

        all_feature_means = torch.concat(all_feature_means, dim=0)  # [d_sae, 2]
        all_feature_single_means = torch.concat(all_feature_single_means, dim=0)  # [d_sae, |reason_tokens|]

        # compute entropy
        probs = all_feature_single_means / (all_feature_single_means.sum(dim=1, keepdim=True) + epsilon)
        log_probs = torch.where(probs > 0, torch.log(probs), 0)
        h = -(probs * log_probs).sum(dim=1)
        # normalize
        if probs.size(1) > 1:
            h_norm = h / math.log(probs.size(1))
        else: # do not compute entropy
            h_norm = torch.ones_like(h)
        
        # compute ReasonScore
        sum_pos = all_feature_means[:, 0].sum(dim=0, keepdim=True) + epsilon
        sum_neg = all_feature_means[:, 1].sum(dim=0, keepdim=True) + epsilon
        
        scores = (
            (all_feature_means[:, 0] / sum_pos) * h_norm**alpha
            - (all_feature_means[:, 1] / sum_neg)
        )

        return scores

    def handle_features(
        self, features: Iterable[int] | None, encoder_wrapper: SAE
    ) -> list[int]:
        if features is None:
            return list(range(encoder_wrapper.cfg.d_sae))
        else:
            return list(features)

    def get_feature_batches(self, features_list: list[int]) -> list[list[int]]:
        # Break up the features into batches
        feature_batches = [
            x.tolist()
            for x in torch.tensor(features_list).split(self.cfg.minibatch_size_features)
        ]
        return feature_batches


def compute_score(
    model_path: str,
    sae_path: str,
    dataset_path: str,
    tokens_str_path: str,
    output_dir: str,
    sae_id: str = None,
    expand_range: Tuple[int, int] = None,
    ignore_tokens: List[int] = None,
    n_samples: int = 4096,
    alpha: float = 1.0,
    column_name: str = "text",
    minibatch_size_features: int = 256,
    minibatch_size_tokens: int = 64,
    num_chunks: int = 1,
    chunk_num: int = 0
):
    """Compute `reasoning` feature scores."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(">>> Loading SAE and LLM")
    if sae_id is None:
        sae = SAE.load_from_pretrained(sae_path, device=device)
    else:
        sae, _, _ = SAE.from_pretrained(sae_path, sae_id, device=device)

    model = HookedTransformer.from_pretrained_no_processing(
        model_path,
        dtype=torch.bfloat16,
        device=device,
    )
    # make pad token different from `bos` and `eos` to prevent removing `bos`/`eos` token during slicing
    if model.tokenizer.pad_token_id == model.tokenizer.eos_token_id:
        model.tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    print(">>> Loading dataset")
    dataset = load_dataset(dataset_path, streaming=False, split="train")
    if column_name == "tokens":
        token_dataset = dataset.shuffle(seed=42)
    else:
        print(">>> Tokenize dataset")
        token_dataset = tokenize_and_concatenate(
            dataset=dataset,
            tokenizer=model.tokenizer,
            streaming=False,
            max_length=sae.cfg.context_size,
            column_name=column_name,
            add_bos_token=sae.cfg.prepend_bos,
            num_proc=4
        ).shuffle(seed=42)
    
    with open(tokens_str_path, 'r') as file:
        tokens_str = json.load(file)

    print(">>> tokens str: {}".format(tokens_str))
    grouped_tokens = defaultdict(list)
    for str_token in tokens_str:
        # we treat " A", "A", " a", "a" as the same token
        normalized_str = str_token.lstrip().lower()
        token_ids = model.tokenizer.encode(str_token, add_special_tokens=False)
        grouped_tokens[normalized_str].append(torch.tensor(token_ids, dtype=torch.long))
    reason_tokens = list(grouped_tokens.values())
    print(">>> tokens ids: {}".format(reason_tokens))
    
    if expand_range is not None:
        print(">>> Using expansion: {}".format(expand_range))
    if ignore_tokens is not None:
        print(">>> Using ignore tokens: {}".format(ignore_tokens))
    
    features = list(range(sae.cfg.d_sae))
    if num_chunks > 1:
        features = split_data(features, num_chunks)[chunk_num]
        print(f">>> Processing features in chunks. Current chunk: {chunk_num}, size: {len(features)}")

    sae_selection_cfg = SaeSelectionConfig(
        hook_point=sae.cfg.hook_name,
        features=features,
        minibatch_size_features=minibatch_size_features,
        minibatch_size_tokens=minibatch_size_tokens,
        device=device,
        dtype="float32"
    )

    feature_scores = SaeSelectionRunner(
        sae_selection_cfg
    ).run(
        encoder=sae,
        model=model,
        tokens=token_dataset["tokens"][:n_samples],
        reason_tokens=reason_tokens,
        ignore_tokens=ignore_tokens,
        expand_range=expand_range,
        alpha=alpha
    )

    # save feature scores
    os.makedirs(output_dir, exist_ok=True)
    output_name = "feature_scores.pt" if num_chunks == 1 else "feature_scores_{}.pt".format(chunk_num)
    torch.save(feature_scores.cpu(), os.path.join(output_dir, output_name))

if __name__ == "__main__":
    fire.Fire(compute_score)
