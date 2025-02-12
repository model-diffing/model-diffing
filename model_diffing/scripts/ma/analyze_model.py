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
import kaleido



import plotly.colors as plc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

layer_names=['blocks.0.hook_resid_pre','blocks.0.hook_resid_mid','blocks.0.hook_resid_post']

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def imshow_fourier(tensor:torch.Tensor, P:int, title='', animation_name='snapshot', facet_labels:Any=[],logged=False, **kwargs:Any):
    fourier_basis,fourier_basis_names=make_fourier_transform(P)
    if tensor.shape[0]==P*P:
        tensor = unflatten_first(tensor,P)
    tensor = torch.squeeze(tensor)
    
    if tensor.dim() == 2:
        #tensor = tensor.unsqueeze(-1)
        pass
    # Original multiple pane version
    if logged:
        tensor = torch.sign(tensor) * torch.log1p(torch.abs(tensor))
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
    
    return fig

def get_activations(model:Transformer,P:int)->Dict[str, Any]:
    all_data = torch.tensor([(i, j, P) for i in range(P) for j in range(P)]).to(device)
    labels = torch.tensor([(i+j)%P for i, j, _ in all_data]).to(device)
    cache = {}
    #model.remove_all_hooks()
    model.cache_all(cache)
    model(all_data)
    model.remove_all_hooks()
    return cache

    
    
    
    
    



def make_neuron_tensor(model:Transformer,P:int,neurons:int=10,layer='blocks.0.mlp.hook_post'):
    neuron_acts=get_activations(model,P)
    neuron_acts = neuron_acts[layer][:, -1]
    neuron_acts=neuron_acts-einops.reduce(neuron_acts, 'batch neuron -> 1 neuron', 'mean')
    top_k=neurons
    fourier_basis, fourier_basis_names = make_fourier_transform(P)
    new_tensor=fft2d(neuron_acts,fourier_basis,P)
    return new_tensor
    


    
    return None

def plot_neuron_k(ft_tensor:torch.Tensor,k:int=8):
    k_tensor=ft_tensor[:,:k]

    imshow_fourier(k_tensor,P,
           title=f'Activations for first {k} neurons in 2D Fourier Basis',
           animation_frame=2,
           animation_name='Neuron')




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
    
    total_energy_flat=total_energy.flatten()

    # Sort values in descending order
    sorted_values, indices = torch.sort(total_energy_flat, descending=True)

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
def freq_num(model:Transformer,P:int,layer='blocks.0.mlp.hook_post',threshold=0.95):
    fourier_rearrange=get_layer_acts(model,P,layer)
    
    #note: 95% worked well on P=97 run - I expect it will work with others, too.
    threshold_indices=find_threshold_indices(fourier_rearrange,threshold)
    print(f'threshold indices shape: {threshold_indices.shape}')

    
    return threshold_indices



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
    

def share_chosen_indices(indices_tensor:torch.Tensor,total_acts_tensor:torch.Tensor,eps=1e-6):

    indexed_tensor=total_acts_tensor[indices_tensor[:,0],indices_tensor[:,1],:]



    indexed_sum=torch.sqrt(torch.sum(total_acts_tensor[indices_tensor[:,0],indices_tensor[:,1],:].pow(2),dim=0))


    total_sum=torch.sqrt(torch.sum(total_acts_tensor.pow(2),dim=(0,1)))
    share_of_sum=(indexed_sum+eps)/(total_sum+eps)
    print(f'indexed sum shape: {indexed_sum.shape}')
    print(f'total sum shape: {total_sum.shape}')
    print(f'share of sum shape: {share_of_sum.shape}')
    
    fig=make_subplots(rows=1,cols=1)
    fig.add_trace(go.Scatter(x=np.arange(0,len(share_of_sum)),y=share_of_sum),row=1,col=1)
    fig.update_yaxes(title_text='Activation share',range=[0,1])
    fig.update_xaxes(title_text='Neuron index')
    fig.update_layout(title_text=f'Share of (L2) activation for chosen {indices_tensor.shape[0]} frequencies')


    return share_of_sum,fig

def vol_pair_plot(model:Transformer,P:int,layers:List=layer_names,power=2):
    
    ft_tensor=sum([get_layer_acts(model,P,layer=layer) for layer in layers[1:]])
    
    squared_abs = (ft_tensor).pow(power)
    if power%2!=0:
        squared_abs = squared_abs.abs()
    

    # For each i,j pair, sum across d dimension
    total_energy = squared_abs.sum(dim=2)
    
    total_energy_flat=total_energy.flatten()

    # Sort values in descending order
    sorted_values, indices = torch.sort(total_energy_flat, descending=True)

    # Find cumulative sum and normalize
    cumsum = torch.cumsum(sorted_values, dim=0)
    cumsum_normalized = cumsum / cumsum[-1]


    fig=make_subplots(rows=1,cols=1)
    fig.add_trace(go.Scatter(x=np.arange(0,len(cumsum_normalized)),y=cumsum_normalized),row=1,col=1)
    fig.update_yaxes(title_text='Cumulative sum of activation volume',range=[0,1])
    fig.update_xaxes(title_text='Fourier pair rank')
    fig.update_layout(title_text=f'Cumulative sum of activation volume L{power} for all neurons in chosen layers')
    #annotate the 95% threshold
    index_95=torch.where(cumsum_normalized>0.95)[0][0]
    # Add vertical line
    fig.add_vline(x=index_95, line_dash='dash', line_color='red', line_width=1)

    # Add annotation for the vertical line
    fig.add_annotation(x=index_95,y=1,text=f'95% act at: {index_95} pairs',showarrow=True,arrowhead=1,ax=0,ay=-40,font=dict(color='red'))

    # Add point highlight at the threshold crossing
    fig.add_trace(
        go.Scatter(
            x=[index_95],
            y=[cumsum_normalized[index_95]],  # Get y-value at threshold
            mode='markers',
            marker=dict(size=10, color='red'),
            showlegend=False,
            name='Threshold Point'
        ),
        row=1, col=1
    )

    # Optional: Add x-axis annotation to highlight the index
    fig.add_annotation(
        x=index_95,
        y=0,  # Place at bottom of plot
        text=f'\n{index_95}',
        showarrow=False,
        yanchor='top',
        font=dict(color='red'),
        yshift=-10  # Offset below x-axis
    )

    return fig




    
        

def collect_model_analysis(model:Transformer,P:int,layers:List=layer_names,save=False,save_dir=''):
    
    freq_nums=torch.unique(torch.cat([freq_num(model,P,layer=layer) for layer in layers]),dim=0)
    
    fig_norm,freqs,freq_names=make_freq_norm_plot(model,P)

        
    
    

    
    #neuron plot
    acts_all_layers=sum([make_neuron_tensor(model,P,layer=layer) for layer in layers])
    acts_all_layers=rearrange_fourier_neel(acts_all_layers,P)

    fig_all_n=imshow_fourier(torch.sum(acts_all_layers**2,dim=-1).sqrt(),P,
          title=f'Activations for all neurons in 2D Fourier Basis',
          animation_frame=2,
          animation_name='Neuron')

    randint=torch.randint(128,(1,)).item()
    
    
    fig_random_neuron=imshow_fourier(acts_all_layers[:,:,randint],P,
          title=f'Activations for random d_model= {randint}, in 2D Fourier Basis',
          animation_frame=2,
          animation_name='Neuron')
    #actvol plot
    share_of_sum,fig_share=share_chosen_indices(freq_nums,acts_all_layers)
    
    
    fig_vol_1=vol_pair_plot(model,P,power=1)
    fig_vol_2=vol_pair_plot(model,P,power=2)
    
    fig_norm.show()
    fig_all_n.show()
    fig_random_neuron.show()
    fig_share.show()
    fig_vol_1.show()
    fig_vol_2.show()


    
    if save:
        save_dir=save_dir+'_'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(save_dir,exist_ok=True)
        fig_norm.write_image(f'{save_dir}/freq_norm_plot.png')
        fig_all_n.write_image(f'{save_dir}/neuron_plot.png')
        fig_random_neuron.write_image(f'{save_dir}/random_neuron_plot.png')
        fig_share.write_image(f'{save_dir}/share_plot.png')
        fig_vol_1.write_image(f'{save_dir}/vol_1_plot.png')
        fig_vol_2.write_image(f'{save_dir}/vol_2_plot.png')
        plot_data={'freqs':freqs,'freq_names':freq_names,'freq_nums':freq_nums}
        torch.save(plot_data,f'{save_dir}/plot_data.pt')


    
    return freq_nums



if __name__=="__main__":
    print("the main character")
    data_dict_path='/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/toy_models2/data/models/113/train_P_113_tf_0.8_lr_0.001_2025-01-24_15-45-50.pt'
    #'/Users/dmitrymanning-coe/Documents/Research/Compact Proofs/code/toy_models2/data/models/97/train_P_97_tf_0.8_lr_0.001_2025-01-24_14-53-36.pt'
    analysis_dir_path='/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/toy_models2/data/analysis/P_113/model_analysis'
    
    model_p=data_dict_path.split('P_')[1].split('_')[0]
    analysis_dir_p=analysis_dir_path.split('P_')[1].split('/')[0]
    assert model_p==analysis_dir_p, f"Model P {model_p} does not match analysis P {analysis_dir_p}"

    
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
    

    #across layer nums:
    collect_model_analysis(model,P,layers=layer_names,save=True,save_dir=analysis_dir_path)
    exit()
    
    all_shapes=[freq_num(model,P,layer=layer) for layer in layer_names]
    print(f'all shapes: {[all_shapes[i].shape for i in range(len(all_shapes))]}')

    cat_shapes=torch.cat(all_shapes,dim=0)
    unique_pairs=torch.unique(cat_shapes,dim=0)
    print(f'unique pairs: {unique_pairs.shape}')
    print(f'unique freqs: {torch.unique(cat_shapes).shape}')

    



    #Done plots
    fig,freq_indices,freq_names=make_freq_norm_plot(model,P)
    #fig.show()
    
    acts_cache=get_activations(model,P)
    print(f'cache keys: {acts_cache.keys()}')
    print(f'resid stream shape reference: {acts_cache["blocks.0.hook_mlp_out"].shape}')
    
    #block.0.hook_resid_pre
    #'blocks.0.attn.hook_attn_pre'
    #'blocks.0.hook_resid_mid'
    #'blocks.0.mlp.hook_post'
    #'blocks.0.hook_resid_post'

    
    freq_no=freq_num(model,P)
    print(f'95% acts no. {freq_no.shape}')
    fig,activation_share,activation_any=neuron_chosenfreq(model,P,freq_indices,neurons=10)
    fig.show()
    

    make_neuron_plot(model,P,neurons=10)
    

    share_tensor=actvol_series(model,P,k=16)
    make_actvol_plot(share_tensor,k=16).show()




    
    
