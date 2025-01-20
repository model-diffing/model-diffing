import torch


def compute_relative_norms(vectors_a: torch.Tensor, vectors_b: torch.Tensor) -> torch.Tensor:
    """Compute relative norms between two sets of vectors."""
    norm_a = torch.norm(vectors_a, dim=-1)
    norm_b = torch.norm(vectors_b, dim=-1)
    return norm_b / (norm_a + norm_b)


def compute_cosine_similarities(vectors_a: torch.Tensor, vectors_b: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarities between corresponding vectors."""
    return torch.nn.functional.cosine_similarity(vectors_a, vectors_b, dim=-1)


def get_shared_latent_mask(
    relative_norms: torch.Tensor, min_thresh: float = 0.3, max_thresh: float = 0.7
) -> torch.Tensor:
    """Create mask for shared latents based on relative norms."""
    return torch.logical_and(relative_norms > min_thresh, relative_norms < max_thresh)
