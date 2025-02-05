import torch
import wandb
from torch.nn.utils import clip_grad_norm_

from model_diffing.analysis.visualization import create_cosine_sim_and_relative_norm_histogram_data
from model_diffing.models.activations.relu import ReLUActivation
from model_diffing.scripts.base_trainer import BaseModelLayerTrainer
from model_diffing.scripts.train_l1_crosscoder.config import L1TrainConfig
from model_diffing.utils import (
    calculate_explained_variance_X,
    calculate_reconstruction_loss,
    get_explained_var_dict,
    l0_norm,
    sparsity_loss_l1_of_norms,
)


class L1CrosscoderTrainer(BaseModelLayerTrainer[L1TrainConfig, ReLUActivation]):
    def _train_step(self, batch_BMLD: torch.Tensor) -> None:
        self.optimizer.zero_grad()

        # fwd
        train_res = self.crosscoder.forward_train(batch_BMLD)

        # losses
        reconstruction_loss = calculate_reconstruction_loss(batch_BMLD, train_res.reconstructed_acts_BXD)
        sparsity_loss = sparsity_loss_l1_of_norms(
            W_dec_HMLD=self.crosscoder.W_dec_HXD,
            hidden_BH=train_res.hidden_BH,
        )
        l1_coef = self._l1_coef_scheduler()
        loss = reconstruction_loss + l1_coef * sparsity_loss

        # backward
        loss.backward()
        clip_grad_norm_(self.crosscoder.parameters(), 1.0)
        self.optimizer.step()
        assert len(self.optimizer.param_groups) == 1, "sanity check failed"
        self.optimizer.param_groups[0]["lr"] = self.lr_scheduler(self.step)

        if (
            self.wandb_run is not None
            and self.cfg.log_every_n_steps is not None
            and self.step % self.cfg.log_every_n_steps == 0
        ):
            mean_l0 = l0_norm(train_res.hidden_BH, dim=-1).mean()

            explained_variance_dict = get_explained_var_dict(
                calculate_explained_variance_X(batch_BMLD, train_res.reconstructed_acts_BXD),
                ("model", list(range(self.n_models))),
                ("layer", self.layers_to_harvest),
            )

            log_dict = {
                "train/l1_coef": l1_coef,
                "train/mean_l0": mean_l0,
                "train/mean_l0_pct": mean_l0 / self.crosscoder.hidden_dim,
                "train/reconstruction_loss": reconstruction_loss.item(),
                "train/sparsity_loss": sparsity_loss.item(),
                "train/loss": loss.item(),
                **explained_variance_dict,
            }

            if self.n_models == 2:
                W_dec_HXD = self.crosscoder.W_dec_HXD.detach().cpu()
                assert W_dec_HXD.shape[1:-1] == (self.n_models, self.n_layers)
                hist_data = create_cosine_sim_and_relative_norm_histogram_data(
                    W_dec_HMLD=W_dec_HXD,
                    layers=self.layers_to_harvest,
                )
                log_dict.update(
                    {f"media/{name}": wandb.Histogram(sequence=data, num_bins=100) for name, data in hist_data.items()}
                )

            self.wandb_run.log(log_dict, step=self.step)

    def _l1_coef_scheduler(self) -> float:
        if self.step < self.cfg.lambda_s_n_steps:
            return self.cfg.lambda_s_max * self.step / self.cfg.lambda_s_n_steps
        else:
            return self.cfg.lambda_s_max
