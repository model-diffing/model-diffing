from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from typing import Any

import tabulate
import torch
from einops import rearrange
from rich import print as rprint
from rich.table import Table
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase

from model_diffing.data.activation_harvester import ActivationsHarvester
from model_diffing.data.token_loader import TokenSequenceLoader
from model_diffing.log import logger
from model_diffing.models.crosscoder import AcausalCrosscoder


@dataclass
class ActivationsWithText:
    activations_BSMPD: torch.Tensor
    tokens_BS: torch.Tensor


@torch.no_grad()
def iterate_activations_with_text(
    token_sequence_loader: TokenSequenceLoader,
    activations_harvester: ActivationsHarvester,
) -> Iterator[ActivationsWithText]:
    for seq in token_sequence_loader.get_sequences_batch_iterator():
        activations_BSMPD = activations_harvester.get_activations_BSMPD(seq.tokens_BS)
        yield ActivationsWithText(activations_BSMPD=activations_BSMPD, tokens_BS=seq.tokens_BS)


@dataclass
class LatentExample:
    tokens_context_S: torch.Tensor
    latents_context_S: torch.Tensor


@dataclass
class LatentSummary:
    selected_examples: list[LatentExample]
    density: float


@torch.no_grad()
def gather_max_activating_examples(
    activations_with_text_iter: Iterator[ActivationsWithText],
    cc: AcausalCrosscoder[Any],
    total_batches: int = 200,  # todo remove me
    context_size: int = 20,
    separation_threshold_tokens: int = 5,
    topk_per_latent_per_batch: int = 50,
    latents_to_inspect: list[int] | None = None,
) -> dict[int, LatentSummary]:
    """
    iterate over the dataloader, gathering context for all non-zero latents, in terms of tokens and latent activations

    returns:
        list of lists of LatentExample, where the outer list is over latents, and the inner list is over activating examples
    """
    if latents_to_inspect is None:
        latents_to_inspect = list(range(cc.hidden_dim))

    examples_by_latent_idx = defaultdict[int, list[LatentExample]](list)
    num_firings_by_latent_idx = defaultdict[int, int](int)

    skipped_firings = 0
    not_skipped_firings = 0

    for activations_with_text in tqdm(
        islice(activations_with_text_iter, total_batches),
        total=total_batches,
        desc="Gathering latent examples",
    ):
        B, S = activations_with_text.activations_BSMPD.shape[:2]
        activations_BsMPD = rearrange(activations_with_text.activations_BSMPD, "b s m p d -> (b s) m p d")
        res = cc.forward_train(activations_BsMPD)
        hidden_BSH = rearrange(res.hidden_BH, "(b s) h -> b s h", b=B, s=S)

        # zero out the activations that are not in the topk
        vals, indices = hidden_BSH.topk(topk_per_latent_per_batch, dim=-1)
        hidden_BSH = torch.zeros_like(hidden_BSH).scatter_(-1, indices, vals)

        for latent_idx in latents_to_inspect:
            for batch_idx in range(B):
                hidden_S = hidden_BSH[batch_idx, :, latent_idx]
                (seq_pos_indices,) = hidden_S.nonzero(as_tuple=True)
                if (hidden_S[seq_pos_indices] < 0.001).any():
                    raise ValueError("fucked up somewhere!")

                start_indices = (seq_pos_indices - context_size).clamp(min=0)
                end_indices = (seq_pos_indices + 1).clamp(max=S)

                last_end = -separation_threshold_tokens
                for start, end in zip(start_indices, end_indices, strict=False):
                    if end < last_end + separation_threshold_tokens:
                        skipped_firings += 1
                    else:
                        not_skipped_firings += 1
                        example = LatentExample(
                            tokens_context_S=activations_with_text.tokens_BS[batch_idx, start:end],
                            latents_context_S=hidden_BSH[batch_idx, start:end, latent_idx],
                        )
                        examples_by_latent_idx[latent_idx].append(example)
                        last_end = end

                assert seq_pos_indices.numel() == (hidden_S > 0).sum().item(), "fucked up somewhere!"
                num_firings_by_latent_idx[latent_idx] += seq_pos_indices.numel()

    total_tokens_seen = total_batches * B * S  # gross, reusing a loop variable, but it works
    logger.info(f"Skipped {skipped_firings} examples, {not_skipped_firings} examples not skipped")

    # within each latent, sort by decreasing activation strength.
    for latent_idx in latents_to_inspect:
        examples_by_latent_idx[latent_idx].sort(key=lambda x: x.latents_context_S[-1].item(), reverse=True)

    return {
        latent_idx: LatentSummary(
            selected_examples=examples_by_latent_idx[latent_idx],
            density=num_firings_by_latent_idx[latent_idx] / total_tokens_seen,
        )
        for latent_idx in latents_to_inspect
    }


def display_latent_examples(latent_examples: list[LatentExample], tokenizer: PreTrainedTokenizerBase):
    for i, example in enumerate(latent_examples):
        print(f"Example {i}")
        print(tokenizer.decode(example.tokens_context_S))
        print(example.latents_context_S)
        print()


def top_and_bottom_logits(
    llm: HookedTransformer,
    sae_W_dec_HD: torch.Tensor,
    latent_indices: list[int] | None = None,
    k: int = 10,
):
    latent_indices_iter = latent_indices if latent_indices is not None else list(range(sae_W_dec_HD.shape[0]))

    for latent_idx in latent_indices_iter:
        logits = sae_W_dec_HD[latent_idx] @ llm.W_U
        pos_logits, pos_token_ids = logits.topk(k)
        pos_tokens = llm.to_str_tokens(pos_token_ids)
        neg_logits, neg_token_ids = logits.topk(k, largest=False)
        neg_tokens = llm.to_str_tokens(neg_token_ids)

        print(
            tabulate.tabulate(
                zip(map(repr, neg_tokens), neg_logits, map(repr, pos_tokens), pos_logits, strict=True),
                headers=["Bottom tokens", "Value", "Top tokens", "Value"],
                tablefmt="simple_outline",
                stralign="right",
                numalign="left",
                floatfmt="+.3f",
            )
        )


def display_top_seqs(data: list[tuple[float, list[str]]], latent_index: int, density: float):
    """
    Given a list of (activation: float, str_toks: list[str]), displays a table of these sequences, with
    the relevant token highlighted.

    We also turn newlines into "\\n", and remove unknown tokens (usually weird quotation marks) for readability.
    """
    table = Table(
        "Act",
        "Sequence",
        title=f"Max Activating Examples (latent {latent_index}, density={(density * 100):.4f}%)",
        show_lines=True,
    )
    for act, str_toks in data:
        formatted_seq = (
            "".join(
                [
                    f"[b u green]{str_tok}[/]" if i == len(str_toks) - 1 else str_tok
                    for i, str_tok in enumerate(str_toks)
                ]
            )
            .replace("�", "")
            .replace("\n", "↵")
        )
        table.add_row(f"{act:.3f}", repr(formatted_seq))
    rprint(table)
