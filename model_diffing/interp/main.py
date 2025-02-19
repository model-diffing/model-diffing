from itertools import islice
from typing import Any

import torch
from einops import pack, rearrange, unpack

from model_diffing.data.model_hookpoint_dataloader import ScaledModelHookpointActivationsDataloader
from model_diffing.models.crosscoder import AcausalCrosscoder


@torch.no_grad()
def gather_max_activations(
    dataloader: ScaledModelHookpointActivationsDataloader,
    cc: AcausalCrosscoder[Any],
    total_batches: int = 200,
    context_size: int = 20,
):
    n_latents = cc.hidden_dim
    latent_examples = [[] for _ in range(n_latents)]
    # activations_with_text.seq.special_tokens_mask_BS
    for activations_with_text in islice(dataloader.iterate_activations_with_text(), total_batches):
        B, S = activations_with_text.activations_BSMPD.shape[:2]
        activations_BsMPD = rearrange(activations_with_text.activations_BSMPD, "b s m p d -> (b s) m p d")
        res = cc.forward_train(activations_BsMPD)
        hidden_BSH = rearrange(res.hidden_BH, "(b s) h -> b s h", b=B, s=S)
        tokens_BS = activations_with_text.seq.tokens_BS

        for latent_idx in range(n_latents):
            latent_BS = hidden_BSH[:, :, latent_idx]
            # padding_BS = torch.ones(B, context_size, device=latent_BS.device) * -1
            # latent_prepad_BS = torch.cat([padding_BS, latent_BS], dim=1)

            batch_indices, seq_pos_indices = latent_BS.nonzero(as_tuple=True)
            seq_pos_start_indices = (seq_pos_indices - context_size).clamp(min=0)

            for firing_batch, firing_seq_pos_start, firing_seq_pos_end in zip(
                batch_indices, seq_pos_start_indices, seq_pos_indices, strict=False
            ):
                tokens_context_S = tokens_BS[firing_batch, firing_seq_pos_start:firing_seq_pos_end]
                latent_context_S = latent_BS[firing_batch, firing_seq_pos_start:firing_seq_pos_end]

                latent_examples[latent_idx].append(
                    {
                        "tokens_context_S": tokens_context_S,
                        "latents_context_S": latent_context_S,
                    }
                )

            # latent_examples[latent_idx].extend(...)


# def get_latents(
#     cc: AcausalCrosscoder[Any],
#     activations_BSMPD: torch.Tensor,
# ) -> torch.Tensor:
#     return hidden_BSH

# t = torch.tensor([[0,

# and the predicted token (?? actually, this is the next token, not the predicted token)
# firing_seq_pos_end = (firing_seq_pos + 1).clamp(max=S)
