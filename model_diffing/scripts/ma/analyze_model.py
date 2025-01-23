import sys
import os
#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from models.ma_transformer import Transformer,TransformerConfig

from dataloader.ma_dataset import datacfg,gen_train_test,get_is_train_test
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


def load_model(data_dict,epoch=None):
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

def make_fourier_transform(P):
    fourier_basis = []
    fourier_basis.append(torch.ones(P)/np.sqrt(P))
    fourier_basis_names = ['Const']
    # Note that if p is even, we need to explicitly add a term for cos(kpi), ie
    # alternating +1 and -1
    for i in range(1, P//2 +1):
        fourier_basis.append(torch.cos(2*torch.pi*torch.arange(P)*i/P))
        fourier_basis.append(torch.sin(2*torch.pi*torch.arange(P)*i/P))
        fourier_basis[-2]/=fourier_basis[-2].norm()
        fourier_basis[-1]/=fourier_basis[-1].norm()
        fourier_basis_names.append(f'cos {i}')
        fourier_basis_names.append(f'sin {i}')
    fourier_basis = torch.stack(fourier_basis, dim=0)

    return fourier_basis, fourier_basis_names


def unflatten_first(tensor,P):
    if tensor.shape[0]==P*P:
        return einops.rearrange(tensor, '(x y) ... -> x y ...', x=P, y=P)
    else:
        return tensor

def rearrange_fourier_neel(tensor,P):
    if tensor.shape[0]==P**2:
        tensor = unflatten_first(tensor,P)
    tensor = torch.squeeze(tensor)
    return tensor

def make_freq_norm_plot(model,P):
    W_E =model.embed.W_E[:, :-1]
    fourier_basis, fourier_basis_names = make_fourier_transform(P)
    norm_fourier=(W_E @ fourier_basis.T).pow(2).sum(0).detach().numpy()

    fig=make_subplots(rows=1,cols=1)
    fig.add_trace(go.Scatter(x=np.arange(0,len(norm_fourier)),y=norm_fourier),row=1,col=1)
    fig.update_xaxes(title_text="Fourier basis index",row=1,col=1)
    fig.update_yaxes(title_text="Norm of Fourier basis projection",row=1,col=1)

    return fig

#bit of a bookie way to do it, because you get a P^2x3 matrix so you need to reshape afterwards
def fft2d(mat,fourier_1d,P):
    # Converts a pxpx... or batch x ... tensor into the 2D Fourier basis.
    # Output has the same shape as the original
    shape = mat.shape
    mat = einops.rearrange(mat, '(x y) ... -> x y (...)', x=P, y=P)
    fourier_mat = torch.einsum('xyz,fx,Fy->fFz', mat, fourier_1d, fourier_1d)
    
    return fourier_mat.reshape(shape)

def imshow_fourier(tensor,P, title='', animation_name='snapshot', facet_labels=[], **kwargs):
    # Set nice defaults for plotting functions in the 2D fourier basis
    # tensor is assumed to already be in the Fourier Basis
    fourier_basis,fourier_basis_names=make_fourier_transform(P)
    if tensor.shape[0]==P*P:
        tensor = unflatten_first(tensor,P)
    tensor = torch.squeeze(tensor)
    fig=px.imshow((tensor).detach().numpy(),
            x=fourier_basis_names,
            y=fourier_basis_names,
            labels={'x':'x Component',
                    'y':'y Component',
                    'animation_frame':animation_name},
            title=title,
            color_continuous_midpoint=0.,
            color_continuous_scale='RdBu',
            **kwargs)
    fig.update(data=[{'hovertemplate':"%{x}x * %{y}y<br>Value:%{z:.4f}"}])
    if facet_labels:
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]['text'] = label
    fig.show()

def get_activations(model,P):
    all_data = torch.tensor([(i, j, P) for i in range(P) for j in range(P)]).to(device)
    labels = torch.tensor([(i+j)%P for i, j, _ in all_data]).to(device)
    cache = {}
    #model.remove_all_hooks()
    model.cache_all(cache)
    model(all_data)
    model.remove_all_hooks()
    return cache

    
    
    
    
    



def make_neuron_plot(model,P,neurons=10):
    neuron_acts=get_activations(model,P)
    neuron_acts = neuron_acts['blocks.0.mlp.hook_post'][:, -1]
    top_k=neurons
    fourier_basis, fourier_basis_names = make_fourier_transform(P)
    imshow_fourier(fft2d(neuron_acts[:, :top_k],fourier_basis,P),P,
           title=f'Activations for first {top_k} neurons in 2D Fourier Basis',
           animation_frame=2,
           animation_name='Neuron')


    
    return None


def top_k_actvol(tensor,k=16):
    s_t=tensor.pow(2)
    f=s_t.shape[0]#fourier components
    n=s_t.shape[-1]#mlp neurons
    st_flat=s_t.view(f * f, n)

    # Sort the squared values in descending order along the first dimension (f*f)
    sorted_vals, _ = torch.sort(st_flat, dim=0, descending=True)

    # Sum the top 16 squared values for each n
    top_16_sum = sorted_vals[:k, :].sum(dim=0)

    # Compute the total sum of squares for each n
    total_sum = st_flat.sum(dim=0)

    # Calculate the share of the top 16 squared values
    share_top_16 = top_16_sum / total_sum

    return share_top_16

def actvol_series(model,P,k=16,eps=1e-6):
    neuron_acts=get_activations(model,P)
    neuron_acts = neuron_acts['blocks.0.mlp.hook_post'][:, -1]
    fourier_basis, fourier_basis_names = make_fourier_transform(P)
    fourier_rearrange=rearrange_fourier_neel(fft2d(neuron_acts,fourier_basis,P),P)

    s_t=fourier_rearrange.pow(2)
    f=s_t.shape[0]#fourier components
    n=s_t.shape[-1]#mlp neurons
    st_flat=s_t.view(f * f, n)

    # Sort the squared values in descending order along the first dimension (f*f)
    sorted_vals, _ = torch.sort(st_flat, dim=0, descending=True)

    # Sum the top 16 squared values for each n
    top_16_sum = sorted_vals[:k, :].sum(dim=0)

    # Compute the total sum of squares for each n
    total_sum = st_flat.sum(dim=0)

    # Calculate the share of the top 16 squared values
    share_top_16 = (top_16_sum+eps) / (total_sum+eps)

    return share_top_16

def make_actvol_plot(share_tensor,k=16):
    fig=make_subplots(rows=1,cols=1)
    k_share=share_tensor.detach().numpy()
    fig.add_trace(go.Scatter(x=np.arange(0,len(k_share)),y=k_share),row=1,col=1)
    fig.update_yaxes(title_text=f'Share of squared activation top {k}',range=[0,1])
    fig.update_xaxes(title_text=f'Neuron')
    fig.update_layout(title_text=f'Share of activation volume for each neuron, top 16 frequencies')
    return fig




if __name__=="__main__":
    print("the main character")
    data_dict_path="/Users/dmitrymanning-coe/Documents/Research/Compact Proofs/code/toy_models2/data/train_P_23_tf_0.8_lr_0.001_2025-01-22_18-09-52.pt"
    data_dict=torch.load(data_dict_path,weights_only=False)

    
    init_model_cfg=data_dict["model_cfg"]
    init_data_cfg=data_dict["data_cfg"]

    P=init_model_cfg.P



    model,state_dic=load_model(data_dict,epoch=None)

    
    

    #get all the labels:

    W_O = einops.rearrange(model.blocks[0].attn.W_O, 'm (i h)->i m h', i=init_model_cfg.n_heads)
    W_K = model.blocks[0].attn.W_K
    W_Q = model.blocks[0].attn.W_Q
    W_V = model.blocks[0].attn.W_V
    W_in = model.blocks[0].mlp.W_in
    b_mlp_in=model.blocks[0].mlp.b_in
    W_out = model.blocks[0].mlp.W_out
    b_out=model.blocks[0].mlp.b_out
    W_pos = model.pos_embed.W_pos.T
    # We remove the equals sign dimension from the Embed and Unembed, so we can
    # apply a Fourier Transform over R^p
    W_E = model.embed.W_E[:, :-1]
    W_U = model.unembed.W_U[:, :-1].T

    # The initial value of the residual stream at position 2 - constant for all inputs
    final_pos_resid_initial = model.embed.W_E[:, -1] + W_pos[:, 2]
    print('W_O', W_O.shape)
    print('W_K', W_K.shape)
    print('W_Q', W_Q.shape)
    print('W_V', W_V.shape)
    print('W_in', W_in.shape)
    print('b_in',b_mlp_in.shape)
    print('W_out', W_out.shape)
    print('W_pos', W_pos.shape)
    print('W_E', W_E.shape)
    print('W_U', W_U.shape)
    print('Initial residual stream value at final pos:', final_pos_resid_initial.shape)

    #make_freq_norm_plot(model,P).show()

    #make_neuron_plot(model,P,neurons=10)

    share_tensor=actvol_series(model,P,k=16)
    make_actvol_plot(share_tensor,k=16).show()




    
    
