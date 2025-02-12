import sys
import os



#sys.path.append('/Users/dmitrymanning-coe/Documents/Research/Compact Proofs/code/toy_models2/code/model-diffing/model_diffing')


import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from model_diffing.models.ma_transformer import Transformer,TransformerConfig

from model_diffing.dataloader.ma_dataset import datacfg,gen_train_test,get_is_train_test
from model_diffing.models.crosscoder import build_relu_crosscoder,build_topk_crosscoder, AcausalCrosscoder
from analyze_model import load_model, get_activations

from model_diffing.scripts.train_topk_crosscoder_light.trainer import TopKTrainer
from model_diffing.scripts.train_l1_crosscoder_light.config import TrainConfig,DecayTo0LearningRateConfig 
from model_diffing.utils import l0_norm, calculate_reconstruction_loss, save_model_and_config, sparsity_loss_l1_of_norms,reduce
from torch.nn.utils import clip_grad_norm_
import copy
from datetime import datetime
from tqdm import tqdm
from typing import TypedDict,Union, Any,List,Dict,Tuple

import plotly.colors as plc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import einops

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import plotly.colors as plc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from collections import defaultdict

#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
def activation_iterator_BMLD(dataset_length,batch_size,train_acts_BMLD):
    while True:
        indices = torch.randint(dataset_length, (batch_size,))

        
        yield train_acts_BMLD[indices]

def vary_hidden(hidden_dims:List,save:bool=False):
    data_dict=defaultdict(dict)

    n_models=1
    n_layers=3
    d_model=128
    dec_init_norm=1#not sure 0.05 worked well before
    lambda_=0
    batch_size = 64
    learning_rate=1e-3
    opt_steps=10_000
    topk=8



    base_config={'n_layers':n_layers,'d_model':d_model,'dec_init_norm':dec_init_norm,'lambda':lambda_,
                 'batch_size':batch_size,'steps':opt_steps,'learning_rate':learning_rate}
    
    data_dict['base_config']=copy.deepcopy(base_config)
    start_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for hidden_dim in tqdm(hidden_dims):
        xcoder=build_topk_crosscoder(n_models, n_layers,d_model,hidden_dim,topk,dec_init_norm)
        
        data_dict[hidden_dim]['xcoder']=xcoder
        data_dict[hidden_dim]['hidden_dim']=hidden_dim

        data_dict_model_path='/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/toy_models2/data/models/113/train_P_113_tf_0.8_lr_0.001_2025-01-24_15-45-50.pt'
        data_dict_model=torch.load(data_dict_model_path,weights_only=False)
        
        model_cfg=data_dict_model["model_cfg"]
        P=model_cfg.P
        data_cfg=data_dict_model["data_cfg"]
        model,state_dict=load_model(data_dict_model)

        activations=get_activations(model,P)

        resid_acts=torch.stack([activations['blocks.0.hook_resid_pre'],activations['blocks.0.hook_resid_mid'],activations['blocks.0.hook_resid_post']],dim=0)
        print(f'residual activations shape {resid_acts.shape}')

        print(f'activations keys {activations.keys()}')

        train_acts_BMLD=einops.rearrange(resid_acts,'layer batch sequence d_model -> (batch sequence) 1 layer d_model')
        dataset_length = train_acts_BMLD.shape[0]

        print(f'train acts rearranged shape {train_acts_BMLD.shape}')




        #so far lr=1e-3, 0.1, l1 coeff=1 and 1000 steps
        train_cfg = TrainConfig(
            learning_rate = DecayTo0LearningRateConfig(
                initial_learning_rate=learning_rate,
                last_pct_of_steps=1.0,
            ),
            num_steps = opt_steps,
            save_dir = '/Users/dmitrymanning-coe/Documents/Research/Compact Proofs/code/toy_models2/data/hidden_sweep',
            save_every_n_steps = opt_steps-1,
            log_every_n_steps = 5_000,
            n_batches_for_norm_estimate =100)


        
        optimizer = torch.optim.Adam(params=xcoder.parameters(), lr=train_cfg.learning_rate.initial_learning_rate,weight_decay=0)


        xc_trainer = TopKTrainer(
            cfg=train_cfg,
            crosscoder=xcoder,
            device=device,
            wandb_run=None,
            optimizer=optimizer,
            dataloader_BMLD=activation_iterator_BMLD(dataset_length,batch_size,train_acts_BMLD),
        )
    

        rec_loss=xc_trainer.train()

        data_dict[hidden_dim]['rec_loss']=rec_loss
        #data_dict[hidden_dim]['sparsity_loss']=sparsity_loss
        


        if save:
            save_dir=f'/Users/dmitrymanning-coe/Documents/Research/Compact Proofs/code/toy_models2/data/hidden_sweep/topk_{topk}'
            os.makedirs(save_dir,exist_ok=True)
            torch.save(data_dict,f'{save_dir}/start_{start_time}')
            print(f'saved to {save_dir}/start_{start_time}.pt')




if __name__=="__main__":
    print("the main character")

    n_models=1
    n_layers=3
    d_model=128
    hidden_dim=200
    dec_init_norm=1#not sure 0.05 worked well before
    lambda_=0
    epochs=300
    batch_size = 64
    topk=4

    vary_hidden([16,17,18,19,20,21,22,23,24,25,30,40,50,60,70,80,90,100,200,300,400,500],save=True)
    exit()
    sweep_path='/Users/dmitrymanning-coe/Documents/Research/Compact Proofs/code/toy_models2/data/hidden_sweep/topk_16/start_2025-01-24 19:07:55'
    sweep_dict=torch.load(sweep_path,weights_only=False)
    hidden_dims=[k for k in sweep_dict if type(k)==int]
    hidden_dims.sort()
    print(f'hidden dims {hidden_dims}')
    rec_losses=[sweep_dict[k]['rec_loss'][-1] for k in hidden_dims]
    fig=make_subplots(rows=1,cols=1)
    fig.add_trace(go.Scatter(x=hidden_dims,y=rec_losses),row=1,col=1)
    fig.update_yaxes(title_text='Reconstruction Loss',type='log',row=1,col=1)
    fig.update_xaxes(title_text='Hidden Dimension')
    fig.update_layout(title_text=f'Hidden Dimension Sweep, topk {topk}')
    fig.show()
    exit()

    xcoder=build_topk_crosscoder(n_models, n_layers,d_model,hidden_dim,topk,dec_init_norm=dec_init_norm)

    
    
    
    data_dict_path='/Users/dmitrymanning-coe/Documents/Research/Compact Proofs/code/toy_models2/data/models/23/train_P_23_tf_0.8_lr_0.001_2025-01-23_09-23-24.pt'
    data_dict=torch.load(data_dict_path,weights_only=False)
    
    model_cfg=data_dict["model_cfg"]
    P=model_cfg.P
    data_cfg=data_dict["data_cfg"]
    model,state_dict=load_model(data_dict)

    activations=get_activations(model,P)

    resid_acts=torch.stack([activations['blocks.0.hook_resid_pre'],activations['blocks.0.hook_resid_mid'],activations['blocks.0.hook_resid_post']],dim=0)
    print(f'residual activations shape {resid_acts.shape}')

    print(f'activations keys {activations.keys()}')

    train_acts_BMLD=einops.rearrange(resid_acts,'layer batch sequence d_model -> (batch sequence) 1 layer d_model')
    dataset_length = train_acts_BMLD.shape[0]

    print(f'train acts rearranged shape {train_acts_BMLD.shape}')


    def activation_iterator_BMLD():
        while True:
            indices = torch.randint(dataset_length, (batch_size,))

            
            yield train_acts_BMLD[indices]

    #so far lr=1e-3, 0.1, l1 coeff=1 and 1000 steps
    train_cfg = TrainConfig(
        learning_rate = DecayTo0LearningRateConfig(
            initial_learning_rate=1e-3,
            last_pct_of_steps=0.9,
        ),
        num_steps = 50_000,
        save_dir = None,#'/Users/dmitrymanning-coe/Documents/Research/Compact Proofs/code/toy_models2/data',
        save_every_n_steps = 100_000,
        log_every_n_steps = 5_000,
        n_batches_for_norm_estimate = 100,
    )

    optimizer = torch.optim.Adam(params=xcoder.parameters(), lr=train_cfg.learning_rate.initial_learning_rate,weight_decay=0)

    xc_trainer = TopKTrainer(
        cfg=train_cfg,
        crosscoder=xcoder,
        device=device,
        wandb_run=None,
        optimizer=optimizer,
        dataloader_BMLD=activation_iterator_BMLD(),
    )
    

    rec_loss=xc_trainer.train()
    xc_dec=xc_trainer.crosscoder.W_dec_HMLD

    def ipr_decoder(tensor:torch.Tensor,r:float=2.0,eps=1e-6):
        rearranged=einops.rearrange(tensor, 'h m l d -> m l (h d)')
        ipr_num=torch.sum(rearranged**(2*r),dim=-1)
        ipr_denom=torch.sum(rearranged**2,dim=-1)**r
        ipr=(ipr_num+eps)/(ipr_denom+eps)
        

        return ipr
        
        

    ipr_test=ipr_decoder(xc_dec)
    print(f'ipr_test shape {ipr_test.shape}')
    print(f'ipr {ipr_test}')

    # fig=make_subplots(rows=1,cols=3)
    # fig.add_trace(go.Heatmap(z=xc_dec[:,0,0,:].detach().numpy()),row=1,col=1)
    # fig.add_trace(go.Heatmap(z=xc_dec[:,0,1,:].detach().numpy()),row=1,col=2)
    # fig.add_trace(go.Heatmap(z=xc_dec[:,0,2,:].detach().numpy()),row=1,col=3)
    # fig.show()

    # fig=make_subplots(rows=1,cols=3)
    # fig.add_trace(go.Histogram(x=(xc_dec[:,0,0,:].detach().numpy()).flatten()),row=1,col=1)
    # fig.add_trace(go.Histogram(x=(xc_dec[:,0,1,:].detach().numpy()).flatten()),row=1,col=2)
    # fig.add_trace(go.Histogram(x=(xc_dec[:,0,2,:].detach().numpy()).flatten()),row=1,col=3)
    # fig.show()

    fig=make_subplots(rows=1,cols=1)
    fig.add_trace(go.Scatter(x=np.arange(0,len(rec_loss)),y=rec_loss,name='Reconstruction loss'),row=1,col=1)
    fig.update_yaxes(title_text='Reconstruction Loss',type='log',row=1,col=1)
    fig.update_xaxes(title_text='Optimization Step')
    fig.update_layout(title_text=f'lr: {train_cfg.learning_rate.initial_learning_rate}, hidden dim: {hidden_dim},topk {topk} dec init norm: {dec_init_norm}')
    fig.show()

    exit()

    # dataset_size = train_acts_BMLD.shape[0]

    # batch_size = 64

    # optimizer = torch.optim.Adam(xcoder.parameters(), lr=5e-5)



    # #in theory, there is a weight between the sparsity penalty and the loss penalty. In theory this has a scheduler, people start the weight at around 0.5 and rampt to 5

    # reconstruction_losses=[]
    # sparsity_losses=[]
    # for epoch in tqdm(range(epochs)):
    #     for step in range(0, dataset_size // batch_size):
    #         optimizer.zero_grad()
    #         batch = train_acts_BMLD[step:step + batch_size]

    #         assert batch.shape == (batch_size, 1, 3, 128)
    #         train_res = xcoder.forward_train(batch)

    #         reconstruction_loss_ = reconstruction_loss(batch, train_res.reconstructed_acts_BMLD)
    #         sparsity_loss_ = sparsity_loss_l1_of_norms(xcoder.W_dec_HMLD, train_res.hidden_BH)
    #         l0_norms_B = reduce(train_res.hidden_BH, "batch hidden -> batch", l0_norm)
    #         l1_coef = lambda_#self._l1_coef_scheduler()
    #         loss = reconstruction_loss_ + l1_coef * sparsity_loss_

    #         if step % 100 == 0:
    #             pass
    #         #print(losses)

    #         loss = reconstruction_loss_ + lambda_ * sparsity_loss_
    #         reconstruction_losses.append(reconstruction_loss_.item())
    #         sparsity_losses.append(sparsity_loss_.item())
    #         loss.backward()
    #         optimizer.step()
    
    # fig=make_subplots(rows=1,cols=2)
    # fig.add_trace(go.Scatter(x=np.arange(0,len(reconstruction_losses)),y=reconstruction_losses,name='Reconstruction loss'),row=1,col=1)
    # fig.add_trace(go.Scatter(x=np.arange(0,len(sparsity_losses)),y=sparsity_losses,name='Sparsity loss'),row=1,col=2)
    # fig.update_yaxes(title_text='Losses',type='log')
    # fig.update_xaxes(title_text='Optimization Step')
    # fig.show()





    # exit()


    

