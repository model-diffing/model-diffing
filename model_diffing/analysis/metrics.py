import torch


# N = num_vectors, F = feature dim
def compute_relative_norms_N(vectors_a_NF: torch.Tensor, vectors_b_NF: torch.Tensor) -> torch.Tensor:
    """Compute relative norms between two sets of vectors."""
    norm_a_N = torch.norm(vectors_a_NF, dim=-1)
    norm_b_N = torch.norm(vectors_b_NF, dim=-1)
    return (norm_b_N + 1e-6) / (norm_a_N + norm_b_N + 1e-6)


def compute_cosine_similarities_N(vectors_a_NF: torch.Tensor, vectors_b_NF: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarities between corresponding vectors."""
    return torch.nn.functional.cosine_similarity(vectors_a_NF, vectors_b_NF, dim=-1)


def get_shared_latent_mask(
    relative_norms: torch.Tensor, min_thresh: float = 0.3, max_thresh: float = 0.7
) -> torch.Tensor:
    """Create mask for shared latents based on relative norms."""
    return torch.logical_and(relative_norms > min_thresh, relative_norms < max_thresh)


def get_IQR_outliers_mask(norms: torch.Tensor, k_iqr: float = 1.5) -> torch.Tensor:
    """Create mask for outliers based on IQR."""
    Q1 = norms.quantile(0.25)
    Q3 = norms.quantile(0.75)
    IQR = Q3 - Q1
    return (norms < (Q1 - k_iqr * IQR)) | (norms > (Q3 + k_iqr * IQR))
