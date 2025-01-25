import sys
import os
#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import TypedDict,Union, Any,List,Dict,Tuple


from model_diffing.models.ma_transformer import Transformer,TransformerConfig

from model_diffing.dataloader.ma_dataset import datacfg,gen_train_test,get_is_train_test
import copy
from datetime import datetime
from tqdm import tqdm
from typing import Any, List, Tuple, Union

import plotly.colors as plc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.graph_objs._figure import Figure
import einops


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import plotly.colors as plc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def load_model(data_dict:Any,epoch:Union[None,int]=None):
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

def make_fourier_transform(P:int)->Tuple[torch.Tensor,List[str]]:
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


def unflatten_first(tensor:torch.Tensor,P:int):
    if tensor.shape[0]==P*P:
        return einops.rearrange(tensor, '(x y) ... -> x y ...', x=P, y=P)
    else:
        return tensor

def rearrange_fourier_neel(tensor:torch.Tensor,P:int):
    if tensor.shape[0]==P**2:
        tensor = unflatten_first(tensor,P)
    tensor = torch.squeeze(tensor)
    return tensor

def make_freq_norm_plot(model:Transformer,P:int):
    W_E =model.embed.W_E[:, :-1]
    fourier_basis, fourier_basis_names = make_fourier_transform(P)
    norm_fourier=(W_E @ fourier_basis.T).pow(2).sum(0).detach().numpy()

    fig=make_subplots(rows=1,cols=1)
    fig.add_trace(go.Scatter(x=np.arange(0,len(norm_fourier)),y=norm_fourier),row=1,col=1)
    fig.update_xaxes(title_text="Fourier basis index",row=1,col=1)
    fig.update_yaxes(title_text="Norm of Fourier basis projection",row=1,col=1)

    sorted_norms=np.sort(norm_fourier)[::-1]
    variance=norm_fourier.var()
    mean=norm_fourier.mean()
    #maybe try everything above the mean
    freq_indices = np.where(norm_fourier > mean)[0]
    freq_components=np.array(fourier_basis_names)[freq_indices]#[fourier_basis_names[i] for i in indices]
    

    return fig,freq_indices,freq_components


#bit of a bookie way to do it, because you get a P^2x3 matrix so you need to reshape afterwards
def fft2d(mat:torch.Tensor,fourier_1d:torch.Tensor,P:int):
    # Converts a pxpx... or batch x ... tensor into the 2D Fourier basis.
    # Output has the same shape as the original
    shape = mat.shape
    mat = einops.rearrange(mat, '(x y) ... -> x y (...)', x=P, y=P)
    fourier_mat = torch.einsum('xyz,fx,Fy->fFz', mat, fourier_1d, fourier_1d)
    
    return fourier_mat.reshape(shape)

def imshow_fourier(tensor:torch.Tensor,P:int, title='', animation_name='snapshot', facet_labels:Any=[], **kwargs:Any):
    # Set nice defaults for plotting functions in the 2D fourier basis
    # tensor is assumed to already be in the Fourier Basis
    fourier_basis,fourier_basis_names=make_fourier_transform(P)
    if tensor.shape[0]==P*P:
        tensor = unflatten_first(tensor,P)
    tensor = torch.squeeze(tensor)

    fig:Figure=px.imshow((tensor).detach().numpy(),
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
            # type: ignore[index]
            fig.layout.annotations[i]['text'] = label
    fig.show()

def get_activations(model:Transformer,P:int)->Dict[str, Any]:
    all_data = torch.tensor([(i, j, P) for i in range(P) for j in range(P)]).to(device)
    labels = torch.tensor([(i+j)%P for i, j, _ in all_data]).to(device)
    cache = {}
    #model.remove_all_hooks()
    model.cache_all(cache)
    model(all_data)
    model.remove_all_hooks()
    return cache

    
    
    
    
    



def make_neuron_plot(model:Transformer,P:int,neurons:int=10,layer='blocks.0.mlp.hook_post'):
    neuron_acts=get_activations(model,P)
    neuron_acts = neuron_acts[layer][:, -1]
    neuron_acts=neuron_acts-einops.reduce(neuron_acts, 'batch neuron -> 1 neuron', 'mean')
    top_k=neurons
    fourier_basis, fourier_basis_names = make_fourier_transform(P)
    imshow_fourier(fft2d(neuron_acts[:, :top_k],fourier_basis,P),P,
           title=f'Activations for first {top_k} neurons in 2D Fourier Basis',
           animation_frame=2,
           animation_name='Neuron')


    
    return None




def top_k_actvol(tensor:torch.Tensor,k:int=16):
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

def actvol_series(model:Transformer,P:int,k=16,eps=1e-6):
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

def make_actvol_plot(share_tensor:torch.Tensor,k=16):
    fig=make_subplots(rows=1,cols=1)
    k_share=share_tensor.detach().numpy()
    fig.add_trace(go.Scatter(x=np.arange(0,len(k_share)),y=k_share),row=1,col=1)
    fig.update_yaxes(title_text=f'Share of squared activation top {k}',range=[0,1])
    fig.update_xaxes(title_text=f'Neuron')
    fig.update_layout(title_text=f'Share of activation volume for each neuron, top 16 frequencies')
    return fig

def merge_consecutive(arr):
   result = []
   i = 0
   while i < len(arr):
       start = arr[i]
       while i + 1 < len(arr) and arr[i+1] - arr[i] == 1:
           i += 1
       result.append(start)
       i += 1
   return result

def get_layer_acts(model:Transformer,P:int,layer='blocks.0.mlp.hook_post'):
    neuron_acts=get_activations(model,P)
    neuron_acts = neuron_acts[layer][:, -1]
    neuron_acts=neuron_acts-einops.reduce(neuron_acts, 'batch neuron -> 1 neuron', 'mean')
    fourier_basis, fourier_basis_names = make_fourier_transform(P)
    fourier_transformed=fft2d(neuron_acts,fourier_basis,P)
    fourier_rearrange=rearrange_fourier_neel(fourier_transformed,P)
    return fourier_rearrange


def find_above_mean_indices(tensor):
    abs_mean = torch.mean(torch.abs(tensor))
    ten_var=tensor.var()
    
    indices_list=[]
    for d in range(tensor.shape[2]):
        d_mean=tensor[:,:,d].abs().mean()
        d_var=tensor[:,:,d].abs().var()
        rows, cols = torch.where(tensor[:,:,d].abs() > abs_mean+2*ten_var)

        indices = torch.stack([rows, cols], dim=1)
        indices_list.append(indices)
    return indices_list

def find_threshold_indices(tensor, threshold=0.95):
   # Get squared absolute values for each position across all d
   squared_abs = torch.square(tensor).abs()
   
   # For each i,j pair, sum across d dimension
   total_energy = squared_abs.sum(dim=2)
   
   # Sort values in descending order
   sorted_values, indices = torch.sort(total_energy.flatten(), descending=True)
   
   # Find cumulative sum and normalize
   cumsum = torch.cumsum(sorted_values, dim=0)
   cumsum_normalized = cumsum / cumsum[-1]
   
   # Get number of pairs needed to reach threshold
   n_pairs = torch.where(cumsum_normalized >= threshold)[0][0] + 1
   
   # Convert flat indices back to 2D coordinates
   selected_indices = indices[:n_pairs]
   rows = selected_indices // tensor.shape[1]
   cols = selected_indices % tensor.shape[1]
   
   return torch.stack([rows, cols], dim=1)


import torch


#The problem for doing threshold indices for each d is that for dead neurons it will essentially be degenerate.
def find_threshold_indices_each_d(tensor, threshold=0.95,eps=1e-6):
   squared_abs = torch.square(tensor).abs()
   indices_list = []
   for d in range(tensor.shape[2]):
       slice_values = squared_abs[:,:,d]
       sorted_values, indices = torch.sort(slice_values.flatten(), descending=True)
       
       cumsum = torch.cumsum(sorted_values+eps, dim=0)
       cumsum_normalized = cumsum / cumsum[-1]
       n_pairs = torch.where(cumsum_normalized >= threshold)[0][0] + 1
       selected_indices = indices[:n_pairs]
       
       rows = selected_indices // tensor.shape[1]
       cols = selected_indices % tensor.shape[1]
       indices_list.append(torch.stack([rows, cols], dim=1))

       
   return indices_list

#I guess you can do an explained variance measure, too.
def freq_num(model:Transformer,P:int,layer='blocks.0.mlp.hook_post'):
    fourier_rearrange=get_layer_acts(model,P,layer)
    
    above_mean=find_above_mean_indices(fourier_rearrange)
    print(f'len above mean: {len(above_mean)}')
    print(f'first entry above mean: {above_mean[0].shape}')
    print(f'first 10 shapes: {[above_mean[i].shape[0] for i in range(10)]}')
    indices_tensor=torch.cat(above_mean)
    print(f'indices tensor shape: {indices_tensor.shape}')
    num_unique_pairs = len(set().union(*[{tuple(sorted(pair.tolist())) for pair in tensor_pairs} for tensor_pairs in above_mean]))
    print(f'unique indices shape: {num_unique_pairs}')
    unique_freqs=torch.unique(indices_tensor)
    print(f'unique freqs shape: {unique_freqs.shape}')

    #threshold
    threshold_indices=find_threshold_indices(fourier_rearrange,threshold=0.95)
    print(f'threshold indices shape: {threshold_indices.shape}')
    print(f'Threshold indices:\n {threshold_indices}')
    exit()
    #28 is exactly right! 4 peaks, 4x4 diags and 2x4 constant cross terms
    threshold_indices_each_d=find_threshold_indices_each_d(fourier_rearrange,threshold=0.95)
    uniques_local=torch.unique(torch.cat(threshold_indices_each_d))
    print(f'unique local freqs shape: {uniques_local.shape}')



    
    
    exit()

    
    return None



def neuron_chosenfreq(model:Transformer,P:int,fourier_freqs:List,neurons:int=10,layer='blocks.0.mlp.hook_post'):
    neuron_acts=get_activations(model,P)
    neuron_acts = neuron_acts[layer][:, -1]
    neuron_acts=neuron_acts-einops.reduce(neuron_acts, 'batch neuron -> 1 neuron', 'mean')
    top_k=neurons
    fourier_basis, fourier_basis_names = make_fourier_transform(P)
    fourier_transformed=fft2d(neuron_acts,fourier_basis,P)
    fourier_rearrange=rearrange_fourier_neel(fourier_transformed,P)
    fourier_indices=[]
    for i in fourier_freqs:
        if i+1 in fourier_freqs:
            fourier_freqs = np.delete(fourier_freqs, np.where(fourier_freqs == i+1))
            fourier_indices.extend([(i,i), (i,i+1),(i+1,i),(i+1,i+1)])
            fourier_indices.extend([(0,i),(0,i+1),(i,0),(i+1,0)])
    print(f'fourier freqs: {fourier_freqs}')
    print(f'fourier indices: {np.array(fourier_indices).shape}')
    fourier_indices = torch.tensor(fourier_indices)
    indexed_tensor=fourier_rearrange[fourier_indices[:, 0], fourier_indices[:, 1], :]
    print(f'indexed tensor shape: {indexed_tensor.shape}')
    
    activation_share=torch.sqrt(torch.sum(indexed_tensor.pow(2),dim=0))/torch.sqrt(torch.sum(fourier_rearrange.pow(2)))
    print(f'activation share shape: {activation_share.shape}')
    
    s_t=fourier_rearrange.pow(2)
    f=s_t.shape[0]#fourier components
    n=s_t.shape[-1]#mlp neurons
    st_flat=s_t.view(f * f, n)

    print(freq_indices)
    
    sorted_indices = torch.argsort(st_flat.pow(2), dim=0, descending=True)  # Sort by squared values
    top_n_indices = sorted_indices[:len(fourier_indices)]  # Take the top n indices for each n
    
    print(torch.unique(top_n_indices).shape)

    print(f'top_n indices shape: {top_n_indices.shape}')
    

    print(f'st_flat top_n shape: {st_flat[torch.unique(top_n_indices)].shape}')
    uniques_sum = torch.sqrt((st_flat[torch.unique(top_n_indices)]**2).sum(dim=1))
    

    # fig=make_subplots(rows=1,cols=1)
    # fig.add_trace(go.Scatter(x=np.arange(0,len(uniques_sum)),y=uniques_sum,name='Normed frequencies'),row=1,col=1)
    # fig.show()
    # exit()

    
    
    



    
    sorted_vals, _ = torch.sort(st_flat, dim=0, descending=True)

    # Sum the top squared values for each n
    print(f'fourier indices shape: {fourier_indices.shape}')
    
    top_n_sum = torch.sqrt(sorted_vals[:len(fourier_indices), :].sum(dim=0))
    
    # Compute the total sum of squares for each n
    total_sum = torch.sqrt(st_flat.sum(dim=0))

    activation_share_any=top_n_sum / total_sum

    fig=make_subplots(rows=1,cols=1)
    fig.add_trace(go.Scatter(x=np.arange(0,len(activation_share)),y=activation_share,name='Normed frequencies'),row=1,col=1)
    fig.add_trace(go.Scatter(x=np.arange(0,len(activation_share_any)),y=activation_share_any,name='Any frequency'),row=1,col=1)
    fig.update_yaxes(title_text=f'Share of squared activation for chosen frequencies',range=[0,1])
    fig.update_xaxes(title_text=f'Neuron index')


    return fig,activation_share,activation_share_any
    
    
    
    
    
    

    imshow_fourier(fft2d(neuron_acts[:, :top_k],fourier_basis,P),P,
           title=f'Activations for first {top_k} neurons in 2D Fourier Basis',
           animation_frame=2,
           animation_name='Neuron')
fourier_indices=[]






if __name__=="__main__":
    print("the main character")
    data_dict_path='/Users/dmitrymanning-coe/Documents/Research/Compact Proofs/code/toy_models2/data/models/97/train_P_97_tf_0.8_lr_0.001_2025-01-24_14-53-36.pt'
    
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
    

    #
    nums_test=freq_num(model,P)
    exit()

    #Done plots
    fig,freq_indices,freq_names=make_freq_norm_plot(model,P)
    fig.show()
    
    acts_cache=get_activations(model,P)
    print(f'cache keys: {acts_cache.keys()}')
    print(f'resid stream shape reference: {acts_cache["blocks.0.hook_mlp_out"].shape}')
    
    #block.0.hook_resid_pre
    #'blocks.0.attn.hook_attn_pre'
    #'blocks.0.hook_resid_mid'
    #'blocks.0.mlp.hook_post'
    #'blocks.0.hook_resid_post'

    

    fig,activation_share,activation_any=neuron_chosenfreq(model,P,freq_indices,neurons=10)
    fig.show()
    

    make_neuron_plot(model,P,neurons=10)
    

    share_tensor=actvol_series(model,P,k=16)
    make_actvol_plot(share_tensor,k=16).show()




    
    
