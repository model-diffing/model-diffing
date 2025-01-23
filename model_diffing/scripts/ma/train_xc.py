import sys
import os




sys.path.append('/Users/dmitrymanning-coe/Documents/Research/Compact Proofs/code/toy_models2/code/model-diffing/model_diffing')


import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from models.ma_transformer import Transformer,TransformerConfig

from models.ma_transformer import Transformer,TransformerConfig
from dataloader.ma_dataset import datacfg,gen_train_test,get_is_train_test
from models.crosscoder import build_relu_crosscoder,build_topk_crosscoder, AcausalCrosscoder
from analyze_model import load_model, get_activations
from scripts.train_l1_crosscoder.trainer import L1CrosscoderTrainer, LossInfo
from utils import l0_norm, reconstruction_loss, save_model_and_config, sparsity_loss_l1_of_norms,reduce
from torch.nn.utils import clip_grad_norm_
import copy
from datetime import datetime
from tqdm import tqdm


import plotly.colors as plc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import einops

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import plotly.colors as plc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px





if __name__=="__main__":
    print("the main character")

    n_models=1
    n_layers=3
    d_model=128
    hidden_dim=1000
    dec_init_norm=0.05#not sure
    lambda_=0.1
    epochs=300

    xcoder=build_relu_crosscoder(n_models, n_layers,d_model,hidden_dim,dec_init_norm)
    
    
    data_dict_path="/Users/dmitrymanning-coe/Documents/Research/Compact Proofs/code/toy_models2/data/train_P_23_tf_0.8_lr_0.001_2025-01-22_18-09-52.pt"
    data_dict=torch.load(data_dict_path,weights_only=False)
    model_cfg=data_dict["model_cfg"]
    P=model_cfg.P
    data_cfg=data_dict["data_cfg"]
    model,state_dict=load_model(data_dict)

    activations=get_activations(model,P)

    resid_acts=torch.stack([activations['blocks.0.hook_resid_pre'],activations['blocks.0.hook_resid_mid'],activations['blocks.0.hook_resid_post']],dim=0)
    print(f'residual activations shape {resid_acts.shape}')

    print(f'activations keys {activations.keys()}')

    train_acts=einops.rearrange(resid_acts,'layer batch sequence d_model -> (batch sequence) 1 layer d_model')

    print(f'train acts rearranged shape {train_acts.shape}')

    #xc_trainer=L1CrosscoderTrainer(dataloader_BMLD=train_acts,cfg=None,llms=[model],crosscoder=AcausalCrosscoder,device=device,wandb_run=None,optimizer=torch.optim.Adam)

    dataset_size = train_acts.shape[0]

    batch_size = 64

    optimizer = torch.optim.Adam(xcoder.parameters(), lr=5e-5)



    #in theory, there is a weight between the sparsity penalty and the loss penalty. In theory this has a scheduler, people start the weight at around 0.5 and rampt to 5

    reconstruction_losses=[]
    sparsity_losses=[]
    for epoch in tqdm(range(epochs)):
        for step in range(0, dataset_size // batch_size):
            optimizer.zero_grad()
            batch = train_acts[step:step + batch_size]

            assert batch.shape == (batch_size, 1, 3, 128)
            train_res = xcoder.forward_train(batch)

            reconstruction_loss_ = reconstruction_loss(batch, train_res.reconstructed_acts_BMLD)
            sparsity_loss_ = sparsity_loss_l1_of_norms(xcoder.W_dec_HMLD, train_res.hidden_BH)
            l0_norms_B = reduce(train_res.hidden_BH, "batch hidden -> batch", l0_norm)
            l1_coef = lambda_#self._l1_coef_scheduler()
            loss = reconstruction_loss_ + l1_coef * sparsity_loss_

            if step % 100 == 0:
                pass
            #print(losses)

            loss = reconstruction_loss_ + lambda_ * sparsity_loss_
            reconstruction_losses.append(reconstruction_loss_.item())
            sparsity_losses.append(sparsity_loss_.item())
            loss.backward()
            optimizer.step()
    
    fig=make_subplots(rows=1,cols=2)
    fig.add_trace(go.Scatter(x=np.arange(0,len(reconstruction_losses)),y=reconstruction_losses,name='Reconstruction loss'),row=1,col=1)
    fig.add_trace(go.Scatter(x=np.arange(0,len(sparsity_losses)),y=sparsity_losses,name='Sparsity loss'),row=1,col=2)
    fig.update_yaxes(title_text='Losses',type='log')
    fig.update_xaxes(title_text='Optimization Step')
    fig.show()





    exit()


    

