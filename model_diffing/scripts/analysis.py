# %%
import torch

torch.set_grad_enabled(False)

# %%
from model_diffing.analysis import metrics
from model_diffing.analysis import visualization

# %%
### Should be replaced by crosscoder decoder vectors
feature_a = torch.randn(100, 128)
feature_b = torch.randn(100, 128)

# %%
relative_norms = metrics.compute_relative_norms(feature_a, feature_b)
relative_norms.shape

# %%
fig = visualization.plot_relative_norms(feature_a, feature_b)
fig.show()

# %%
shared_latent_mask = metrics.get_shared_latent_mask(relative_norms)
shared_latent_mask.shape

# %%
cosine_sims = metrics.compute_cosine_similarities(feature_a, feature_b)
cosine_sims.shape

# %%
shared_features_cosine_sims = cosine_sims[shared_latent_mask]
fig = visualization.plot_cosine_sim(shared_features_cosine_sims)
fig.show()
