import torch
from model_diffing.models.ma_transformer import Transformer,TransformerConfig
from typing import Any,Union,Dict,Any,Iterator
from einops import reduce
from einops.einops import Reduction
from functools import partial
from model_diffing.scripts.config_common import BaseExperimentConfig, BaseTrainConfig
from pathlib import Path
from torch import nn
import yaml #type: ignore
from model_diffing.log import logger
from tqdm import tqdm
from itertools import islice
import einops
import sys


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_neuron_preacts_cutoff(enc_acts_BH:torch.Tensor,W_dec_PHD:torch.Tensor,b_dec_PD:torch.Tensor,W_ins:torch.Tensor,b_ins:torch.Tensor,W_outs:torch.Tensor,b_outs:torch.Tensor,device:str="cpu",bias:float=0):
    """
    Idea of this calculation is to cutoff the encoding past the point
    where the contributions are negligible.
    This is actually not great - you should do it via the non-zero indexing...
    """

    hidden_dim=W_dec_PHD.shape[1]
    #enc_acts_BH=enc_acts_BH.to(device)
    #W_dec_PHD=W_dec_PHD.to(device)
    #b_dec_PD=b_dec_PD.to(device)
    #W_ins=W_ins.to(device)
    #b_ins=b_ins.to(device)
    #W_outs=W_outs.to(device)
    #b_outs=b_outs.to(device)

    #So first thing we need to do is to sort the enc_acts_BH by the absolute value of the features
    
    sorted_enc_vals,sorted_enc_inds=torch.sort(torch.abs(enc_acts_BH),dim=-1,descending=True)
    
    #b_dec_PBHD=b_dec_PD[:,sorted_enc_inds,:]
    
    
    # Find the largest index in dim=1 that is not zero for each element in dim=0
    #Ah that's clever - because non zero elements are ones you can just sum
    #to ge the largest value!
    
    #Note - hopefully, this filtering step is cheap, so you can always do it first
    #and then you can do filtering on the pushed through activations, too
    non_zero_indices = (sorted_enc_vals != 0).sum(dim=1)
    # Get the maximum index across all elements in dim=0
    max_non_zero_index = non_zero_indices.max().item()

    filtered_sorted_enc_BH=sorted_enc_vals[:,:max_non_zero_index]
    
    
    #That doesn't make any sense?
    print(f'shape W_dec_PHD {W_dec_PHD.shape}')
    
    W_dec_PBHD=W_dec_PHD[:,sorted_enc_inds[:,:max_non_zero_index],:]
    #filtered_W_dec_PHD=W_dec_PHD[:,:max_non_zero_index,:]
    
    
    enc_BHD_W = einops.einsum(filtered_sorted_enc_BH[...,None], W_dec_PBHD, "batch hidden_c one, block batch hidden_c d_model -> block batch d_model hidden_c")
    #print(f'enc_BHD_W.shape: {enc_BHD_W.shape}')
    enc_BHD_b = bias*b_dec_PD[:,None,:,None]/hidden_dim
    #print(f'enc_BHD_b.shape: {enc_BHD_b.shape}')
    p_BNH = einops.einsum(W_ins, enc_BHD_W+enc_BHD_b, "block d_model d_mlp, block batch d_model hidden -> block batch d_mlp hidden")
    #print(f'p_BNH.shape: {p_BNH.shape}')
    #print(f'b_ins.shape: {b_ins.shape}')
    p_BNH += bias*b_ins[:,None,:,None]/hidden_dim
    #print(f'p_BNH.shape: {p_BNH.shape}')
    #OK, now I want to reindex the cutoff indices to the original indices
    
    
    return p_BNH,sorted_enc_inds

def mlp_preacts_simple(enc_acts_BH:torch.Tensor,W_dec:torch.Tensor,b_dec:torch.Tensor,W_in:torch.Tensor,b_in:torch.Tensor,bias:float=1.0):
    """
    This is the simple way to do it - just use the pre-activations of the encoder.
    """
    
    W_in_W_dec=W_in@W_dec.T
    W_in_b_dec=W_in@b_dec
        
    preacts_BNH=W_in_W_dec[None,:,:]*enc_acts_BH[:,None,:]
    preacts_BNH+=W_in_b_dec[None,:,None]+b_in[None,:,None]
    
    return preacts_BNH


def mlp_preacts_2(enc_acts_BH:torch.Tensor,W_dec:torch.Tensor,b_dec:torch.Tensor,W_in:torch.Tensor,b_in:torch.Tensor,bias:float=1.0):
    
    W_dec_W_in = W_dec @ W_in.T
    b_dec_W_in = b_dec @ W_in.T
    
    batch_size,hidden_dim=W_dec.shape
    d_mlp=W_in.shape[1]
    
    preacts_BNH=torch.zeros(batch_size,d_mlp,hidden_dim)#feature_activations_SH = crosscoder._encode_BH(activations_SMLD)

    # enc_acts_BH  : (B, F)   – sparse feature activations (F = # features)
# W_dec_W_in   : (F, H)   – weight that maps each feature → d_mlp-dim pre-activation
# b_dec_W_in   : (H,)     – bias that belongs with W_dec_W_in
# b_in         : (H,)     – optional extra bias (drop if you don’t need it)

    B, F = enc_acts_BH.shape
    H     = W_dec_W_in.shape[1]
    
    
    device = enc_acts_BH.device
    dtype  = enc_acts_BH.dtype

    # 1) Find all non-zero locations once
    batch_idx, feat_idx = (enc_acts_BH != 0).nonzero(as_tuple=True)        # (N,)

    # 2) Pull the corresponding scalar activations
    scales = enc_acts_BH[batch_idx, feat_idx]                              # (N,)

    # 3) Compute the per-feature pre-activations only for active features
    #    (broadcast: (N,1)  *  (N,H)  →  (N,H))
    preacts_sparse = (
        W_dec_W_in[feat_idx] * scales.unsqueeze(1) +        # feature-specific term
        b_dec_W_in + b_in                                   # constant term
    )                                                       # (N, H)

    # 4) Scatter them back into a dense (B, F, H) tensor
    preacts_BFH = torch.zeros(B, F, H, device=device, dtype=dtype)
    preacts_BFH[batch_idx, feat_idx] = preacts_sparse       # preserves indices

    
    # for s in range(batch_size):
    #     active_features = torch.where(enc_acts_BH[s, :] != 0.0)[0]
    #     per_feature_preactivations_HA = W_dec_W_in[active_features, :] * enc_acts_BH[s, active_features, None]+bias*b_dec_W_in[None,None,:]+bias*b_in[None,None,:]
    #     preacts_BNH[s,active_features,:]=per_feature_preactivations_HA

    print(f'preacts_BNH.shape: {preacts_BNH.shape}')
    raise Exception('Stop here')
    
    # active_features=torch.where(enc_acts_BH!=0.0)[0]
    # per_feature_preactivations_HA = W_dec_W_in[active_features, :] * enc_acts_BH[None,active_features,None]+bias*b_dec_W_in[None,None,:]+bias*b_in[None,None,:]
    # print(f'per_feature_preactivations_HA.shape: {per_feature_preactivations_HA.shape}')
    # raise Exception('Stop here')
    return per_feature_preactivations_HA

"""
First want to get an activations tensor from the model.
Would be helpful to have to keep it in the structure of the xcoder.
"""

def save_model_and_config(config: BaseTrainConfig, save_dir: Path, model: nn.Module, step: int) -> None:
    """Save the model to disk. Also save the config file if it doesn't exist.

    Args:
        config: The config object. Saved if save_dir / "config.yaml" doesn't already exist.
        save_dir: The directory to save the model and config to.
        model: The model to save.
        step: The current step (used in the model filename).
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    if not (save_dir / "config.yaml").exists():
        with open(save_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)
        logger.info("Saved config to %s", save_dir / "config.yaml")

    model_file = save_dir / f"model_step_{step}.pt"
    torch.save(model.state_dict(), model_file)
    logger.info("Saved model to %s", model_file)

def l0_norm(
    input: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    out: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return torch.norm(input, p=0, dim=dim, keepdim=keepdim, out=out, dtype=dtype)


def l1_norm(
    input: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    out: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return torch.norm(input, p=1, dim=dim, keepdim=keepdim, out=out, dtype=dtype)


def l2_norm(
    input: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    out: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return torch.norm(input, p=2, dim=dim, keepdim=keepdim, out=out, dtype=dtype)


def _weighted_l1_sparsity_loss(
    W_dec_HMLD: torch.Tensor,
    hidden_BH: torch.Tensor,
    layer_reduction: Reduction,  # type: ignore
    model_reduction: Reduction,  # type: ignore
) -> torch.Tensor:
    assert (hidden_BH >= 0).all()
    # think about it like: each latent (called "hidden" here) has a separate projection onto each (model, layer)
    # so we have a separate l2 norm for each (hidden, model, layer)
    W_dec_l2_norms_HML = reduce(W_dec_HMLD, "hidden model layer dim -> hidden model layer", l2_norm)

    # to get the weighting factor for each latent, we reduce it's decoder norms for each (model, layer)
    reduced_norms_H = multi_reduce(
        W_dec_l2_norms_HML,
        "hidden model layer",
        ("layer", layer_reduction),
        ("model", model_reduction),
    )

    # now we weight the latents by the sum of their norms
    weighted_hiddens_BH = hidden_BH * reduced_norms_H
    weighted_l1_of_hiddens_BH = reduce(weighted_hiddens_BH, "batch hidden -> batch", l1_norm)
    return weighted_l1_of_hiddens_BH.mean()


sparsity_loss_l2_of_norms = partial(
    _weighted_l1_sparsity_loss,
    layer_reduction=l2_norm,
    model_reduction=l2_norm,
)

sparsity_loss_l1_of_norms = partial(
    _weighted_l1_sparsity_loss,
    layer_reduction=l1_norm,
    model_reduction=l1_norm,
)


def calculate_reconstruction_loss(activation_BMLD: torch.Tensor, target_BMLD: torch.Tensor) -> torch.Tensor:
    """This is a little weird because we have both model and layer dimensions, so it's worth explaining deeply:

    The reconstruction loss is a sum of squared L2 norms of the error for each activation space being reconstructed.
    In the Anthropic crosscoders update, they don't write for the multiple-model case, they write it as:

    $$\\sum_{l \\in L} \\|a^l(x_j) - a^{l'}(x_j)\\|^2$$

    Here, I'm assuming we want to expand that sum to be over models, so we would have:

    $$ \\sum_{m \\in M} \\sum_{l \\in L} \\|a_m^l(x_j) - a_m^{l'}(x_j)\\|^2 $$
    """
    error_BMLD = activation_BMLD - target_BMLD
    error_norm_BML = reduce(error_BMLD, "batch model layer d_model -> batch model layer", l2_norm)
    squared_error_norm_BML = error_norm_BML.square()
    summed_squared_error_norm_B = reduce(squared_error_norm_BML, "batch model layer -> batch", torch.sum)
    return summed_squared_error_norm_B.mean()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# (oli) sorry - this is probably overengineered
def multi_reduce(
    tensor: torch.Tensor,
    shape_pattern: str,
    *reductions: tuple[str, Reduction],  # type: ignore
) -> torch.Tensor:
    original_shape = einops.parse_shape(tensor, shape_pattern)
    for reduction_dim, reduction_fn in reductions:
        if reduction_dim not in original_shape:
            raise ValueError(f"Dimension {reduction_dim} not found in original_shape {original_shape}")
        target_pattern_pattern = shape_pattern.replace(reduction_dim, "")
        exec_pattern = f"{shape_pattern} -> {target_pattern_pattern}"
        shape_pattern = target_pattern_pattern
        tensor = reduce(tensor, exec_pattern, reduction_fn)

    return tensor


def calculate_explained_variance_ML(
    activations_BMLD: torch.Tensor,
    reconstructed_BMLD: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """for each model and layer, calculate the mean explained variance inside each d_model feature space"""
    error_BMLD = activations_BMLD - reconstructed_BMLD

    mean_error_var_ML = error_BMLD.var(-1).mean(0)
    mean_activations_var_ML = activations_BMLD.var(-1).mean(0)

    explained_var_ML = 1 - (mean_error_var_ML / (mean_activations_var_ML + eps))
    return explained_var_ML


def get_explained_var_dict(explained_variance_ML: torch.Tensor, layers_to_harvest: list[int]) -> dict[str, float]:
    num_models, _n_layers = explained_variance_ML.shape
    explained_variances_dict = {
        f"train/explained_variance/M{model_idx}_L{layer_number}": explained_variance_ML[model_idx, layer_idx].item()
        for model_idx in range(num_models)
        for layer_idx, layer_number in enumerate(layers_to_harvest)
    }

    return explained_variances_dict


def get_decoder_norms_H(W_dec_HMLD: torch.Tensor) -> torch.Tensor:
    W_dec_l2_norms_HML = reduce(W_dec_HMLD, "hidden model layer dim -> hidden model layer", l2_norm)
    norms_H = reduce(W_dec_l2_norms_HML, "hidden model layer -> hidden", torch.sum)
    return norms_H


def size_GB(tensor: torch.Tensor) -> float:
    return tensor.numel() * tensor.element_size() / (1024**3)




@torch.no_grad()
def estimate_norm_scaling_factor_ML(
    dataloader_BMLD: Iterator[torch.Tensor],
    device: torch.device,
    n_batches_for_norm_estimate: int,
) -> torch.Tensor:
    d_model = next(dataloader_BMLD).shape[-1]
    mean_norms_ML = _estimate_mean_norms_ML(dataloader_BMLD, device, n_batches_for_norm_estimate)
    scaling_factors_ML = torch.sqrt(torch.tensor(d_model)) / mean_norms_ML
    return scaling_factors_ML

@torch.no_grad()
# adapted from SAELens https://github.com/jbloomAus/SAELens/blob/6d6eaef343fd72add6e26d4c13307643a62c41bf/sae_lens/training/activations_store.py#L370
def _estimate_mean_norms_ML(
    dataloader_BMLD: Iterator[torch.Tensor],
    device: torch.device,
    n_batches_for_norm_estimate: int,
) -> torch.Tensor:
    norm_samples = []

    for batch_BMLD in tqdm(
        islice(dataloader_BMLD, n_batches_for_norm_estimate),
        desc="Estimating norm scaling factor",
        total=n_batches_for_norm_estimate,
    ):
        batch_BMLD = batch_BMLD.to(device)
        norms_means_ML = multi_reduce(
            batch_BMLD,
            "batch model layer d_model",
            ("d_model", l2_norm),
            ("batch", torch.mean),
        )
        norm_samples.append(norms_means_ML)

    norm_samples_NML = torch.stack(norm_samples, dim=0)
    mean_norms_ML = reduce(norm_samples_NML, "batch model layer -> model layer", torch.mean)
    return mean_norms_ML


@torch.no_grad()
def collect_norms(
    dataloader_BMLD: Iterator[torch.Tensor],
    device: torch.device,
    n_batches: int,
) -> torch.Tensor:
    norm_samples = []

    for batch_BMLD in tqdm(
        islice(dataloader_BMLD, n_batches),
        desc="Collecting norms",
        total=n_batches,
    ):
        batch_BMLD = batch_BMLD.to(device)
        norms_BML = reduce(batch_BMLD, "batch model layer d_model -> batch model layer", l2_norm)
        norm_samples.append(norms_BML)

    norm_samples_NML = torch.cat(norm_samples, dim=0)
    return norm_samples_NML


def load_model(data_dict:Any,epoch:Union[None,int]=None):
    model_cfg=data_dict["model_cfg"]
    data_cfg=data_dict["data_cfg"]

    model = Transformer(model_cfg)
    model.to(device)

    if epoch==None:
        epochs_saved=[k for k in data_dict.keys() if type(k)==int]
        epoch=max(epochs_saved)
    state_dict=data_dict[epoch]['model']
    model.load_state_dict(state_dict)
    return model,state_dict

def get_activations(model:Transformer,P:int)->Dict[str, Any]:
    all_data = torch.tensor([(i, j, P) for i in range(P) for j in range(P)]).to(device)
    labels = torch.tensor([(i+j)%P for i, j, _ in all_data]).to(device)
    cache = {}
    #model.remove_all_hooks()
    model.cache_all(cache)
    model(all_data)
    model.remove_all_hooks()
    return cache
