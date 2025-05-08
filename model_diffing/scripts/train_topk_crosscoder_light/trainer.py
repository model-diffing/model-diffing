from collections.abc import Iterator

import torch
import wandb
from einops import einsum
from torch.nn.utils import clip_grad_norm_
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase
from wandb.sdk.wandb_run import Run

from model_diffing.log import logger
from model_diffing.models.crosscoder_light import AcausalCrosscoder
from model_diffing.scripts.train_topk_crosscoder_light.config import TrainConfig,DecayTo0LearningRateConfig
#from model_diffing.scripts.utils import estimate_norm_scaling_factor_ML
from model_diffing.scripts.utils import estimate_norm_scaling_factor_X
from model_diffing.utils import calculate_reconstruction_loss
from model_diffing.scripts.ma.utils import save_model_and_config
from tqdm import tqdm
from model_diffing.scripts.ma.utils import get_neuron_preacts_cutoff,mlp_preacts_2,mlp_preacts_simple
from einops import reduce
from model_diffing.utils import l2_norm
from transformer_lens import HookedTransformer,utils
import torch.nn as nn
import sys
from model_diffing.scripts.ma.utils import get_activations
import einops


# ['blocks.0.hook_resid_pre',
# 'blocks.0.attn.hook_k',
# 'blocks.0.attn.hook_q', 
# 'blocks.0.attn.hook_v', 
# 'blocks.0.attn.hook_attn_pre',
# 'blocks.0.attn.hook_attn',
# 'blocks.0.attn.hook_z', 
# 'blocks.0.hook_attn_out',
# 'blocks.0.hook_resid_mid',
# 'blocks.0.mlp.hook_pre',
# 'blocks.0.mlp.hook_post',
# 'blocks.0.hook_mlp_out',
# 'blocks.0.hook_resid_post']

def get_im_penalty(crosscoder:AcausalCrosscoder,model:HookedTransformer,enc_acts_BH:torch.Tensor)->torch.Tensor:
    
    W_in=model.blocks[0].mlp.W_in
    W_out=model.blocks[0].mlp.W_out
    b_in=model.blocks[0].mlp.b_in
    b_out=model.blocks[0].mlp.b_out

    
    
    # mlp_resid_mid=raw_acts_BMLD[:,0,1,:]/estimate_norm_scaling_factors_ML[0,1]
    # mlp_resid_post=raw_acts_BMLD[:,0,2,:]/estimate_norm_scaling_factors_ML[0,2]
    
    
    # mlp_pre    = nn.functional.linear(mlp_resid_mid,W_in,b_in)    # (B, d_mlp)
    # mlp_hidden = nn.ReLU()(mlp_pre)                  # (B, d_mlp)
    # mlp_out    = nn.functional.linear(mlp_hidden,W_out,b_out)      # (B, D)
    # recon_resid_post = mlp_resid_mid + mlp_out     # (B, D)

    # # now check
    # err = torch.norm(recon_resid_post - mlp_resid_post) / torch.norm(mlp_resid_post)
    # print(f"reconstruction error: {100*err:.1f}%")  # → near 0
    
    # sys.exit()

    W_dec=crosscoder.W_dec_HMLD
    #model is really batch here
    W_dec=einops.rearrange(W_dec,'hidden_c model layer d_model -> model layer hidden_c d_model')
    b_dec=crosscoder.b_dec_MLD
    
    
    
    #mlp_preacts=get_neuron_preacts_cutoff(reconstructed_acts_BMLD,W_dec,b_dec,W_in,b_in,W_out,b_out,"cpu",bias=1)
    
    mlp_preacts=mlp_preacts_simple(enc_acts_BH,W_dec[0,1,:,:],b_dec[0,1,:],W_in,b_in)
    
    # feat_max_inds=torch.max(mlp_preacts.abs(),dim=-1).indices
    # feat_max_vals=mlp_preacts[feat_max_inds]
    feat_max=torch.max(mlp_preacts.abs(),dim=-1).values
    minus_max=(mlp_preacts.abs()).sum(dim=-1)-feat_max
    mean_minus_max=minus_max.mean()
    return mean_minus_max



class TopKTrainer:
    def __init__(
        self,
        cfg: TrainConfig,
        #llms: list[HookedTransformer],
        optimizer: torch.optim.Optimizer,
        dataloader_BMLD: Iterator[torch.Tensor],
        crosscoder: AcausalCrosscoder,
        wandb_run: Run | None,
        device: torch.device,
        model: HookedTransformer,
        lambda_im_penalty: float,
    ):
        self.cfg = cfg
        #self.llms = llms

        # assert all(llm.tokenizer == llms[0].tokenizer for llm in llms), (
        #     "All models must have the same tokenizer"
        # )
        #tokenizer = self.llms[0].tokenizer
        #assert isinstance(tokenizer, PreTrainedTokenizerBase)
        #self.tokenizer = tokenizer
        self.crosscoder = crosscoder
        self.optimizer = optimizer
        self.dataloader_BMLD = dataloader_BMLD
        self.wandb_run = wandb_run
        self.device = device
        self.d_model=next(iter(self.dataloader_BMLD)).shape[-1]

        self.step = 0
        self.model=model
        self.lambda_im_penalty=lambda_im_penalty
        

    # @property
    # def d_model(self) -> int:
    #     d_model_=next(iter(self.dataloader_BMLD)).shape[-1]
    #     return d_model_

    def train(self):
        rec_loss=[]
        penalty_loss=[]
        
        logger.info("Estimating norm scaling factors (model, layer)")
        norm_scaling_factors_ML = self._estimate_norm_scaling_factor_ML()
        logger.info(f"Norm scaling factors (model, layer): {norm_scaling_factors_ML}")

        if self.wandb_run:
            wandb.init(
                project=self.wandb_run.project,
                entity=self.wandb_run.entity,
                config=self.cfg.model_dump(),
            )
        pbar = tqdm(total=self.cfg.num_steps, desc="Training")
        while self.step < self.cfg.num_steps:
            
            batch_BMLD = self._next_batch_BMLD(norm_scaling_factors_ML)
            log_dict = self._train_step(batch_BMLD)
            rec_loss.append(log_dict['train/reconstruction_loss'])
            penalty_loss.append(log_dict['train/im_penalty'])

            if self.wandb_run and (self.step + 1) % self.cfg.log_every_n_steps == 0:
                self.wandb_run.log(log_dict)

            if self.cfg.save_dir and self.cfg.save_every_n_steps and (self.step + 1) % self.cfg.save_every_n_steps == 0:
                save_model_and_config(
                    config=self.cfg,
                    save_dir=self.cfg.save_dir,
                    model=self.crosscoder,
                    step=self.step,
                )

            self.step += 1
            pbar.update(1)
            pbar.set_description(f"Rec Loss: {log_dict['train/reconstruction_loss']:.4f} Penalty Loss: {log_dict['train/im_penalty']:.4f}")
        
        return rec_loss,penalty_loss
    
    

    def _train_step(self, batch_BMLD: torch.Tensor) -> dict[str, float]:
        self.optimizer.zero_grad()

        #There's no sparsity penalty here?
        train_res = self.crosscoder.forward_train(batch_BMLD)
        loss = calculate_reconstruction_loss(train_res.reconstructed_acts_BMLD,batch_BMLD)
        #self._estimate_norm_scaling_factor_ML()
        im_penalty=get_im_penalty(self.crosscoder,self.model,train_res.hidden_BH)
        
        
        #loss_, loss_info = self._get_loss(batch_BMLD)

        penalized_loss=loss+self.lambda_im_penalty*im_penalty
        penalized_loss.backward()
        clip_grad_norm_(self.crosscoder.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.param_groups[0]["lr"] = self._lr_scheduler()

        log_dict = {
            "train/step": self.step,
            "train/reconstruction_loss": loss.item(),
            "train/im_penalty": im_penalty.item(),
            "train/penalized_loss": penalized_loss.item(),
        }

        return log_dict

    def _estimate_norm_scaling_factor_ML(self) -> torch.Tensor:
        return estimate_norm_scaling_factor_X(
            self.dataloader_BMLD,
            self.device,
            self.cfg.n_batches_for_norm_estimate,
        )

    def _next_batch_BMLD(self, norm_scaling_factors_ML: torch.Tensor) -> torch.Tensor:
        batch_BMLD = next(self.dataloader_BMLD)
        batch_BMLD = batch_BMLD.to(self.device)
        batch_BMLD = einsum(
            batch_BMLD, norm_scaling_factors_ML, "batch model layer d_model, model layer -> batch model layer d_model"
        )
        return batch_BMLD

    def _lr_scheduler(self) -> float:
        pct_until_finished = 1 - (self.step / self.cfg.num_steps)
        if pct_until_finished < self.cfg.learning_rate.last_pct_of_steps:
            # 1 at the last step of constant learning rate period
            # 0 at the end of training
            scale = pct_until_finished / self.cfg.learning_rate.last_pct_of_steps
            return self.cfg.learning_rate.initial_learning_rate * scale
        else:
            return self.cfg.learning_rate.initial_learning_rate
