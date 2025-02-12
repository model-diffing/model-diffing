import sys
import os



#sys.path.append('/Users/dmitrymanning-coe/Documents/Research/Compact Proofs/code/toy_models2/code/model-diffing/model_diffing')


import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


from model_diffing.utils import calculate_reconstruction_loss
from model_diffing.models.ma_transformer import Transformer,TransformerConfig

from model_diffing.dataloader.ma_dataset import datacfg,gen_train_test,get_is_train_test
from model_diffing.models.crosscoder import build_relu_crosscoder,build_topk_crosscoder, AcausalCrosscoder
from analyze_model import load_model, get_activations,make_fourier_transform,rearrange_fourier_neel,fft2d,imshow_fourier,freq_num
from model_diffing.scripts.train_l1_crosscoder_light.trainer import L1CrosscoderTrainer, LossInfo
from model_diffing.scripts.train_l1_crosscoder_light.config import TrainConfig, DecayTo0LearningRateConfig
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

from utils import get_activations,load_model


layer_names=['blocks.0.hook_resid_pre','blocks.0.hook_resid_mid','blocks.0.hook_resid_post']

def cross_entropy_high_precision(logits:torch.Tensor, labels:torch.Tensor):
    # Shapes: batch x vocab, batch
    # Cast logits to float64 because log_softmax has a float32 underflow on overly
    # confident data and can only return multiples of 1.2e-7 (the smallest float x
    # such that 1+x is different from 1 in float32). This leads to loss spikes
    # and dodgy gradients
    logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1)
    loss = -torch.mean(prediction_logprobs)
    return loss

def errors_histogram(raw_acts:torch.Tensor,rec_acts:torch.Tensor,random_samples:int=10_000):
    """
    Make the figure of histogram of errors between raw and reconstructed activations
    
    Args:
        raw_acts: Raw model activations tensor
        rec_acts: Reconstructed activations tensor from crosscoder
        random_samples: Number of random samples to use for histogram
        
    Returns:
        Plotly figure showing histograms of activation magnitudes and reconstruction errors
    """
    rec_error = (raw_acts-rec_acts).abs()+1e-6
    random_samples_idx = np.random.randint(0,rec_error.shape[0],random_samples)
    
    fig = make_subplots(rows=1,cols=1)
    
    # Get activation values and their mean
    
    
    act_vals = np.log10(np.abs(((raw_acts+1e-6)[random_samples_idx].detach().numpy()).ravel()))
    act_mean = 10**np.mean(act_vals)
    
    # Get reconstruction error values and their mean 
    rec_vals = np.log10((rec_error[random_samples_idx].detach().numpy()).ravel())
    rec_mean = 10**np.mean(rec_vals)
    
    fig.add_trace(go.Histogram(x=act_vals, name='activations'), row=1, col=1)
    fig.add_trace(go.Histogram(x=rec_vals, name='reconstruction_error'), row=1, col=1)
    
    # Add vertical lines at means
    fig.add_vline(x=np.log10(act_mean), line_dash="dash", line_color="blue", 
                    annotation_text=f"Raw Activation Mean: {act_mean:.2e}", annotation_position="top")
    fig.add_vline(x=np.log10(rec_mean), line_dash="dash", line_color="red",
                    annotation_text=f"Reconstruction Error Mean: {rec_mean:.2e}", annotation_position="top")
    
    fig.update_xaxes(title_text=f"Activation Magnitude/Reconstruction Error (log10, {random_samples} samples)")
    fig.update_yaxes(title_text="Count")
    
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.5)
    
    return fig

def volume_sum(model_acts:torch.Tensor, # shape: (batch*seq, 1, layers, d_model)
               rec_acts:torch.Tensor):   # shape: (batch*seq, 1, layers, d_model)
    
    errors=(model_acts-rec_acts).abs()
    flatten_acts=model_acts.detach().numpy().ravel()
    percent_error=errors/(model_acts.abs())
    percent_error_array=percent_error.detach().numpy().ravel()
    sorted_perror_ind=np.argsort(percent_error_array)
    sorted_perror_array=percent_error_array[sorted_perror_ind]

        
    acts_error_ordered=np.abs(flatten_acts[sorted_perror_ind])

    acts_error_ordered = acts_error_ordered.astype(np.float64)


    #Technically this is less reliable, but I'll keep it this way
    #because if there's a gap it shows you that there is a numerical precision
    #issue
    cum_acts=np.cumsum(acts_error_ordered)/np.sum(acts_error_ordered)
    gap=1-cum_acts[-1]
    if gap>0.01:
        print(f'Gap is large:, {gap}.  Consider increasing precision')

    fig=make_subplots(rows=1,cols=1)
    fig.add_trace(go.Scatter(x=sorted_perror_array[::1000],y=cum_acts[::1000]),row=1,col=1)
    
    # Find index closest to 1% error
    closest_idx = np.abs(sorted_perror_array - 0.01).argmin()
    share_below_1pct = cum_acts[closest_idx]
    
    fig.add_vline(x=0.01, line_width=3, line_dash="dash", line_color="black",
                 annotation_text=f"{share_below_1pct:.1%} of activation volume below 1% error", annotation_position="top")
    fig.update_xaxes(title_text='Percentage error',range=[0,0.05],tickformat='.0%')
    fig.update_yaxes(title_text='Cumulative activation volume',tickformat='.0%')
    return fig

def get_raw_and_rec_acts(model: Transformer, data_cfg: datacfg, xc:AcausalCrosscoder,layers=layer_names) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get raw model activations and their reconstructions from the crosscoder
    
    Args:
        model: The transformer model to get activations from
        data_cfg: Data config containing parameters like P
        xc: The trained crosscoder
        
    Returns:
        Tuple of (raw_activations, reconstructed_activations) with shape (batch*seq, layers, d_model)
    """
    
    # Get raw activations from model
    acts = get_activations(model, data_cfg.P)
    
    resid_acts = torch.stack([acts[layer_name] for layer_name in layers], dim=0)
    
    # Reshape for crosscoder input (batch*seq, 1, layers, d_model)
    train_acts_BMLD = einops.rearrange(resid_acts, 
        'layer batch sequence d_model -> (batch sequence) 1 layer d_model')
    # The inverse operation would be:
    # resid_acts = einops.rearrange(train_acts_BMLD,
    #     '(batch sequence) 1 layer d_model -> layer batch sequence d_model')
    
    # Get reconstructions
    hidden = xc.encode(train_acts_BMLD)
    reconstructed = xc.decode(hidden)
    
    
    return train_acts_BMLD, reconstructed,hidden

class ActivationReplacer:
    def __init__(self, model):
        """Initialize with a transformer model."""
        self.model = model
        self.replacement_hooks = {}
        self.hook_handles = []
        
    def register_replacement(self, hook_name, activation_tensor):
        """Register a tensor to replace activations at a specific hook point."""
        self.replacement_hooks[hook_name] = activation_tensor
        
    def _create_replacement_hook(self, hook_name):
        """Create a hook function that replaces activations with registered tensor."""
        def hook_fn(activation, name):
            if name in self.replacement_hooks:
                # Ensure shapes match
                replacement = self.replacement_hooks[name]
                if replacement.shape != activation.shape:
                    raise ValueError(
                        f"Shape mismatch at {name}. "
                        f"Expected {activation.shape}, got {replacement.shape}"
                    )
                return replacement
            return activation
        return hook_fn
    
    def apply(self):
        """Apply all registered replacements to the model."""
        # Remove any existing hooks first
        self.remove()
        
        # Add new hooks for each registered replacement
        for hook_point in self.model.hook_points():
            if hook_point.name in self.replacement_hooks:
                handle = hook_point.add_hook(
                    self._create_replacement_hook(hook_point.name)
                )
                self.hook_handles.append(handle)
    
    def remove(self):
        """Remove all replacement hooks."""
        for hook_point in self.model.hook_points():
            hook_point.remove_hooks('fwd')
        self.hook_handles = []

# Example usage:
# replacer = ActivationReplacer(transformer_model)
# replacer.register_replacement('blocks.0.hook_attn_out', autoencoder_tensor)
# replacer.apply()
# output = transformer_model(input_tokens)
# replacer.remove()

def reconstruction_loss(model:Transformer,rec_acts:torch.Tensor,replaced_layers:List[str]):
    model_inserted=copy.deepcopy(model)
    model_original=copy.deepcopy(model)

    replacer=ActivationReplacer(model_inserted)

    rec_tensor_reversed = einops.rearrange(rec_acts,
    '(batch sequence) 1 layer d_model -> layer batch sequence d_model', sequence=3)

    for layer_name in replaced_layers:
        layer_idx=layer_names.index(layer_name)
        replacer.register_replacement(layer_name,rec_tensor_reversed[layer_idx])

    replacer.apply()

    all_data = torch.tensor([(i, j, P) for i in range(P) for j in range(P)]).to(device)
    labels = torch.tensor([(i+j)%P for i, j, _ in all_data]).to(device)

    inserted_logits = model_inserted(all_data)[:, -1]
    inserted_logits = inserted_logits[:, :-1]
    inserted_loss = cross_entropy_high_precision(inserted_logits, labels)

    original_logits = model_original(all_data)[:, -1]
    original_logits = original_logits[:, :-1]
    original_loss = cross_entropy_high_precision(original_logits, labels)

    replacer.remove()

    return inserted_loss.item(), original_loss.item()

def plot_loss_ratios_by_layer(model:Transformer,rec_acts:torch.Tensor):
    layers_inserted=[]
    loss_ratios=[]
    for layer_name in layer_names:
        inserted_loss, original_loss = reconstruction_loss(model,rec_acts,[layer_name])
        layers_inserted.append(layer_name)
        loss_ratios.append(inserted_loss/original_loss)

    fig = make_subplots(rows=1, cols=1)

    # Create bar chart
    fig.add_trace(
        go.Bar(
            name='Loss Ratio',
            x=layers_inserted,
            y=loss_ratios,
            text=[f'Penalty from rec.:{ratio-1:.3%}' for ratio in loss_ratios],
            textposition='auto'
        )
    )

    # Update layout
    fig.update_layout(
        title='Loss Ratio (Reconstructed/Original) by Layer',
        xaxis_title='Layer',
        yaxis_title='Loss Ratio of model with reconstructed activations vs. original',
        barmode='group',
        showlegend=True
    )

    # Add horizontal line at y=1 to show baseline
    fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                annotation_text="Original Loss", 
                annotation_position="right")
    
    fig.update_yaxes(tickformat='.1%',range=[0.95,2.0])

    return fig



def imshow_fourier(tensor, title='', animation_name='snapshot', facet_labels=[],P:int=113, **kwargs):
    # Set nice defaults for plotting functions in the 2D fourier basis
    # tensor is assumed to already be in the Fourier Basis
    if tensor.shape[0]==P*P:
        tensor = unflatten_first(tensor)
    tensor = torch.squeeze(tensor)
    fig=px.imshow(tensor.detach().numpy(),
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


def imshow_fourier2(tensor, title='', facet_labels=[], P:int=113, **kwargs):
    # Set nice defaults for plotting functions in the 2D fourier basis
    # tensor is assumed to already be in the Fourier Basis
    if tensor.shape[0] == P*P:
        tensor = unflatten_first(tensor)
    tensor = torch.squeeze(tensor)
    
    # Create frames for each slice in the d dimension
    frames = []
    for i in range(tensor.shape[2]):  # Iterate through the d dimension
        frames.append(tensor[:,:,i])
    
    # Convert to numpy and create a list of frame dictionaries
    frame_data = [{"data": tensor[:,:,i].detach().numpy(), "name": f"frame_{i}"} 
                  for i in range(tensor.shape[2])]
    
    not_fourier=True
    if not_fourier:
        x_labels=None
        y_labels=None
    fig = px.imshow(frames[0].detach().numpy(),  # Show first frame initially
            #x=fourier_basis_names,
            #y=fourier_basis_names,
            labels={'x':'x Component',
                    'y':'y Component'},
            title=title,
            color_continuous_midpoint=0.,
            color_continuous_scale='RdBu',
            **kwargs)
            
    # Add slider
    sliders = [{
        'currentvalue': {'prefix': 'Slice: '},
        'steps': [
            {
                'method': 'animate',
                'args': [[f'frame_{i}'], {
                    'mode': 'immediate',
                    'frame': {'duration': 0, 'redraw': True},
                    'transition': {'duration': 0}
                }],
                'label': str(i)
            } for i in range(tensor.shape[2])
        ]
    }]
    
    # Update figure with frames and slider
    fig.frames = [go.Frame(
        data=[go.Heatmap(z=frame_data[i]["data"])],
        name=f'frame_{i}'
    ) for i in range(tensor.shape[2])]
    
    fig.update_layout(sliders=sliders)
    fig.update(data=[{'hovertemplate':"%{x}x * %{y}y<br>Value:%{z:.4f}"}])
    
    if facet_labels:
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]['text'] = label
            
    fig.show()

# Example usage:
# tensor = torch.randn(113, 113, 10)  # Example tensor with 10 frames
# imshow_fourier_2(tensor).show()

def diagonal_features_plot(f_hidden_acts:torch.Tensor,P:int):
    diagonals_above_mean=f_hidden_acts[np.arange(f_hidden_acts.shape[0]),np.arange(f_hidden_acts.shape[0]),:]
    print(f'diagonals_above_mean shape {diagonals_above_mean.shape}')

    # Create heatmap of diagonal values
    fig = go.Figure(data=go.Heatmap(
        z=diagonals_above_mean.detach().numpy(),
        colorscale='RdBu',
        zmid=0
    ))

    # Update layout with axis labels
    fig.update_layout(
        xaxis_title="Feature",
        yaxis_title="Frequency",
        title="Diagonal Values Heatmap"
    )

    fig.show()

def pairs_features_plot(model:Transformer,f_hidden_acts:torch.Tensor,P:int):

    threshold_indices=freq_num(model,P)

    print(f'threshold_indices shape {threshold_indices.shape}')

    threshold_features=f_hidden_acts[threshold_indices[:,0],threshold_indices[:,1],:]

    print(f'threshold_features shape {threshold_features.shape}')

    # Create heatmap of threshold features
    fig = go.Figure(data=go.Heatmap(
        z=threshold_features.detach().numpy(),
        colorscale='RdBu',
        zmid=0
    ))

    # Update layout with axis labels
    fig.update_layout(
        xaxis_title="Feature",
        yaxis_title="Frequency pairs",
        title="Threshold Features Heatmap"
    )

    fig.show()


if __name__ == '__main__':
    print('the main character')

    saved_model_path='/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/toy_models2/data/models/113/train_P_113_tf_0.8_lr_0.001_2025-01-24_15-45-50.pt'
    #train_xc_path='/Users/dmitrymanning-coe/Documents/Research/Compact Proofs/code/toy_models2/data/hidden_sweep/summarydicts/113/start_2025-02-04 08:28:46'
    train_xc_path='/Users/dmitrymanning-coe/Documents/Research/Compact Proofs/code/toy_models2/data/hidden_sweep/summarydicts/113/start_2025-02-06 12:50:41'
    xc_dict=torch.load(train_xc_path,weights_only=False)
    

    #Need to implement some checks on the loaded data to make sure P's are right etc..
    
    
    print(f'xc_dict keys {xc_dict.keys()}')
    
    xc=xc_dict[50]["xcoder"]
    data_dict=torch.load(saved_model_path,weights_only=False)
    model,state_dict=load_model(data_dict)

    data_cfg=data_dict["data_cfg"]
    P=data_cfg.P

    fourier_basis, fourier_basis_names = make_fourier_transform(P)
    print(f'fourier_basis shape {fourier_basis.shape}')
    
    acts=get_activations(model,P)
    print(f'acts keys {acts.keys()}')
    
    exit()


    raw_acts,rec_acts,hidden_acts=get_raw_and_rec_acts(model,data_cfg,xc)
    

    print(f'raw_acts shape {raw_acts.shape}')
    raw_acts_mid=raw_acts[:,0,1,:]
    raw_acts_mid=raw_acts_mid - einops.reduce(raw_acts_mid, 'batch neuron -> 1 neuron', 'mean')
    raw_acts_mid=einops.rearrange(raw_acts_mid,'(batch sequence) model -> batch sequence model', sequence=3)
    raw_acts_mid_f=fft2d(raw_acts_mid,fourier_basis,P)
    raw_acts_mid_f=rearrange_fourier_neel(raw_acts_mid_f,P)
    raw_acts_mid_f_rearrange=rearrange_fourier_neel(raw_acts_mid_f,P)
    raw_acts_mid_rearrange=rearrange_fourier_neel(raw_acts_mid,P)

    
    print(f'raw_acts_mid_f_rearrange shape {raw_acts_mid_f_rearrange.shape}')
    
    print(f'raw_acts_mid_rearrange shape {raw_acts_mid_rearrange.shape}')
    imshow_fourier2(raw_acts_mid_f_rearrange[:,:,-1,50:])
    exit()
    mid_summed_seq=einops.reduce(raw_acts_mid_rearrange[-1].abs(),'f1 f2 neuron -> neuron', 'sum')
    mid_summed_seq_f=einops.reduce(raw_acts_mid_f_rearrange[-1].abs(),'f1 f2 neuron -> neuron', 'sum')

    # Create a plotly figure for comparing summed sequences
    fig = go.Figure()
    
    # Add the spatial domain trace
    fig.add_trace(go.Scatter(
        y=mid_summed_seq,
        name='Spatial Domain',
        mode='lines'
    ))
    
    # Add the Fourier domain trace 
    fig.add_trace(go.Scatter(
        y=mid_summed_seq_f,
        name='Fourier Domain',
        mode='lines'
    ))
    
    # Update layout
    fig.update_layout(
        title='Summed Sequences: Spatial vs Fourier Domain',
        xaxis_title='Neuron Index',
        yaxis_title='Summed Magnitude',
        showlegend=True
    )
    
    fig.show()
    
    exit()


    rec_loss=calculate_reconstruction_loss(raw_acts,rec_acts)

    print(f'rec_loss {rec_loss}')

    error_fig=errors_histogram(raw_acts,rec_acts,random_samples=1_000)
    #error_fig.show()

    vol_fig=volume_sum(raw_acts,rec_acts)
    #vol_fig.show()

    #exit()

    rec_loss_fig=plot_loss_ratios_by_layer(model,rec_acts)
    #rec_loss_fig.show()
    
    

    #Let's try to do the Fourier things!


    print(f'hidden_acts shape {hidden_acts.shape}')
    
    invert_hidden_acts=einops.rearrange(hidden_acts,'(batch sequence) hidden -> batch sequence hidden', sequence=3)
    invert_hidden_acts = invert_hidden_acts - einops.reduce(invert_hidden_acts, 'batch sequence neuron -> batch 1 neuron', 'mean')


    
    exit()
    
    fourier_transformed=fft2d(invert_hidden_acts,fourier_basis,P)
    
    fourier_rearrange=rearrange_fourier_neel(fourier_transformed,P)
    
    

    print(f'fourier_rearrange shape {fourier_rearrange.shape}')

    f_hidden_acts=fourier_rearrange[:,:,-1,:]
    flattened=rearrange_fourier_neel(invert_hidden_acts,P)
    
    

    #imshow_fourier2(f_hidden_acts[:,:,:100])

    
    #f_hidden_acts=fourier_rearrange[:,:,2,:]
    #pairs_features_plot(model,f_hidden_acts,P)
    #diagonal_features_plot(f_hidden_acts,P)
    


    #Let's now try to look at the ReLU interaction metric

    
    print(f'raw_acts shape {raw_acts.shape}')

    #model weights
    W_O = einops.rearrange(model.blocks[0].attn.W_O, 'm (i h)->i m h', i=model.cfg.n_heads)
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
    

    def manual_layer(x, W_in, W_out, b_mlp_in, b_out):
        # MLP computation
        with torch.no_grad():
            pre_acts = einops.einsum(W_in, x, "d_mlp d_model, token d_model -> token d_mlp") + b_mlp_in
            post_acts = nn.ReLU()(pre_acts)
            out_acts = einops.einsum(W_out, post_acts, "d_model d_mlp, token d_mlp -> token d_model") + b_out

        return pre_acts,post_acts,out_acts

    #First, we need to carry out a correctness check:
    #will resid_pre_relu + mlp_output = resid_post_relu?

    print(f'raw_acts shape {raw_acts.shape}')
    acts_dic=get_activations(model,data_cfg.P)
    print(f'acts_dic keys {acts_dic.keys()}')
    
    
    pre_mlp=raw_acts[:,0,1,:]
    post_mlp=raw_acts[:,0,2,:]

    manual_mlp_pre,manual_mlp_post,manual_mlp_out=manual_layer(pre_mlp,W_in,W_out,b_mlp_in,b_out)
    print(f'manual mlp correct? {torch.allclose(manual_mlp_out+pre_mlp,post_mlp)}')
    
    enc_acts=xc.encode(raw_acts)
    dec_acts=xc.decode(enc_acts)

    feat_mlp_pre,feat_mlp_post,feat_mlp_out=manual_layer(dec_acts[:,0,1,:],W_in,W_out,b_mlp_in,b_out)

    #manual decoding
    activation_BMLD = einops.einsum(enc_acts,xc.W_dec_HMLD,"batch hidden, hidden model layer d_model -> batch model layer d_model")
    activation_BMLD += xc.b_dec_MLD
    print(f'activation_BMLD shape {activation_BMLD.shape}')
    print(f'manual decoding correct? {torch.allclose(dec_acts,activation_BMLD)}')
    print(f'enc_acts shape {enc_acts.shape}')
    

    def get_one_feature_contribution(enc_acts:torch.Tensor,xc:AcausalCrosscoder,feature_index:int):
        """
        To understand this just think about the matrix - we're going to put each of the terms
        in the sum that makes up an entry into a seperate dimension.
        To recover the [p^2,d] matrix, we just need to sum over the feature dimension.
        """

        one_feature_dec=einops.einsum(enc_acts[:,feature_index].unsqueeze(1),xc.W_dec_HMLD,"batch hidden, hidden model layer d_model -> batch model layer d_model")
        one_feature_dec += xc.b_dec_MLD
        
        one_mlp_pre,one_mlp_post,one_mlp_out=manual_layer(one_feature_dec[:,0,1,:],W_in,W_out,b_mlp_in,b_out)
        # print(f'one_mlp_pre shape {one_mlp_pre.shape}')
        # print(f'one_mlp_post shape {one_mlp_post.shape}')
        # print(f'one_mlp_out shape {one_mlp_out.shape}')

        one_mlp_pre=einops.rearrange(one_mlp_pre,'(batch sequence) hidden -> batch sequence hidden', sequence=3)
    
    

        one_mlp_pre_rearrange=rearrange_fourier_neel(one_mlp_pre,P)
        #print(f'one_mlp_pre_rearrange shape {one_mlp_pre_rearrange.shape}')

        one_mlp_f=fft2d(one_mlp_pre,fourier_basis,P)
        #print(f'one_mlp_f shape {one_mlp_f.shape}')
        one_mlp_f_reshape=rearrange_fourier_neel(one_mlp_f,P)
        #print(f'one_mlp_f_reshape shape {one_mlp_f_reshape.shape}')
        return one_mlp_pre_rearrange,one_mlp_f_reshape
    
    def comp_f_oneneuron_onefeature(one_mlp_pre_rearrange:torch.Tensor,one_mlp_f_reshape:torch.Tensor):
        # Get data and find shared color range
        data1 = one_mlp_pre_rearrange[:,:,2,0].detach().numpy()
        data2 = one_mlp_f_reshape[:,:,2,0].detach().numpy()
        vmin = min(data1.min(), data2.min())
        vmax = max(data1.max(), data2.max())
    
        fig=make_subplots(rows=1,cols=2,subplot_titles=['Comp','Fourier'])
        
        fig.add_trace(go.Heatmap(z=data1, colorscale='RdBu', zmin=vmin, zmax=vmax, showscale=False),row=1,col=1)
        fig.add_trace(go.Heatmap(z=data2, colorscale='RdBu', zmin=vmin, zmax=vmax),row=1,col=2)
        fig.update_layout(title='One Feature, One Neuron, Comp. vs Fourier')
        
        return fig

    
    #one_mlp_pre_rearrange,one_mlp_f_reshape=get_one_feature_contribution(enc_acts,xc,0)
    #comp_f_oneneuron_onefeature(one_mlp_pre_rearrange,one_mlp_f_reshape).show()

    #First contstruct the tensor of features, and verify that summing them gives you the correct matrix

    #Another approach to constructing the feature tensor - just don't contract the decoding sum!
    #dec_neuron_layer=xc.W_dec_HMLD[:,:,1,0]
    dec_nocontract = enc_acts[:,:,None] * xc.W_dec_HMLD[None,:,0,1,:]
    
    dec_nocontract=einops.rearrange(dec_nocontract,'batch hidden model -> batch model hidden')
    print(f'dec_nocontract shape {dec_nocontract.shape}')
   
    
    #Let's invert so we can ony look at last token. The more scalable solution is to just load chunks
    #to the hard drive and then reload them since you don't need the whole matrix for anything.
    #But this is engineering effort, so let's just get preliminary results this way first -
    #it's not a fundamental limitation.

    dec_nocontract=einops.rearrange(dec_nocontract,'(batch sequence) model hidden -> batch sequence model hidden', sequence=3)
    dec_nocontract_last=dec_nocontract[:,-1,:,:]
    
    print(f'dec_nocontract_last shape {dec_nocontract_last.shape}')
    
    dec_contract_last=dec_nocontract_last.sum(dim=-1)
    decoded=einops.einsum(enc_acts,xc.W_dec_HMLD,"batch hidden, hidden model layer d_model -> batch model layer d_model")
    decoded=decoded[:,0,1,:]
    print(f'dec contract last shape {dec_contract_last.shape}')
    print(f'decoded shape {decoded.shape}')
    invert_decoded=einops.rearrange(decoded,'(batch sequence) hidden -> batch sequence hidden', sequence=3)
    invert_decoded_last=invert_decoded[:,-1,:]
    
    print(f'invert_decoded_last shape {invert_decoded_last.shape}')
    print(f'dec contract last shape {dec_contract_last.shape}')
    print(f'inv decoded shape {invert_decoded_last.shape}')
    
    print(f'dec contract correct? {torch.sum(dec_contract_last-invert_decoded_last)}')
    
    #I think you'd have to share the bias between the features
    dec_nocontract_bias=dec_nocontract_last+xc.b_dec_MLD[None,0,1,:,None]/dec_nocontract_last.shape[-1]

    print(f'Correct with bias? {torch.sum(dec_nocontract_bias.sum(dim=-1)-invert_decoded_last-xc.b_dec_MLD[None,0,1,:])}')
    #I think that's right with some extra error from the floating point division
    

    #Now, that the reconstruction works, we just need to multiply that tensor by the in matrix of the mlp
    #First, we need to get the in matrix of the mlp
    #Urgh, seems this is too big to do in one tensor step? Yep, unfortunately, will need a for loop equivalent...

    print(f'W_in shape {W_in.shape}')
    print(f'dec_nocontract_bias shape {dec_nocontract_bias.shape}')
    
    pre_acts = einops.einsum(W_in, dec_nocontract_bias,
                        "d_mlp d_model, token d_model hidden -> token d_mlp hidden")
    pre_acts+=b_mlp_in.unsqueeze(0).unsqueeze(2)
    print(f'pre_acts shape {pre_acts.shape}')

    print(f'pre_acts shape {pre_acts.shape}')
    
    def get_preacts_nocontract(enc_acts:torch.Tensor,xcoder:AcausalCrosscoder,layer_index:int=1,token_index:Union[int,None]=-1):
        """
        Get the preacts without contracting the feature dimension
        """
        print(f'enc_acts shape {enc_acts.shape}')
        
        #Note that you have to divide by the number of features to get the correct bias
        dec_nocontract = enc_acts[:,:,None] * xc.W_dec_HMLD[None,:,0,layer_index,:]+xc.b_dec_MLD[0,layer_index]/enc_acts.shape[1]
        dec_nocontract=einops.rearrange(dec_nocontract,'(batch sequence) hidden d_model -> batch sequence hidden d_model', sequence=3)
        print(f'dec_nocontract shape {dec_nocontract.shape}')
        dec_nocontract_last=dec_nocontract[:,-1,:,:]
        print(f'dec_nocontract_last shape {dec_nocontract_last.shape}')
        check_contract=True
        if check_contract:
            dec_contract=xcoder.decode(enc_acts)
            dec_contract=einops.rearrange(dec_contract[:,0,1,:],'(batch sequence) d_model -> batch sequence d_model', sequence=3)
            dec_contract_last=dec_contract[:,-1,:]
            print(f'dec_contract_last shape {dec_contract_last.shape}')
            print(f'dec_nocontract_last shape {dec_nocontract_last.shape}')
            
            print(f'dec_contract_last correct? {100*torch.sum(dec_contract_last-dec_nocontract_last.sum(dim=1))/torch.sum(dec_contract_last)}% error')

        #OK so decoding is correct, then let's push through the mlp
        pre_acts=einops.einsum(W_in,dec_nocontract_last,"d_mlp d_model, batch hidden d_model -> batch hidden d_mlp")
        pre_acts+=b_mlp_in/dec_nocontract_last.shape[1]
        pre_acts=einops.rearrange(pre_acts,'batch hidden d_mlp -> batch d_mlp hidden')
        
        
        return pre_acts
            
        
    
    
    pre_acts=get_preacts_nocontract(enc_acts,xc,token_index=None)
    i_1=torch.sum(torch.sum(pre_acts.abs(),dim=-1),dim=0)
    i_inf=torch.sum(torch.sum(pre_acts,dim=-1).abs(),dim=0)

    def form_i2(pre_acts:torch.Tensor):

        i_2_tensor=torch.zeros(pre_acts.shape[1],pre_acts.shape[-1],pre_acts.shape[-1])
        i_2_nonint_tensor=torch.zeros(pre_acts.shape[1],pre_acts.shape[-1],pre_acts.shape[-1])
        print(f'i_2_tensor shape {i_2_tensor.shape}')
        print(f'i_2_nonint shape {i_2_nonint_tensor.shape}')
        for neuron_ind in tqdm(range(pre_acts.shape[1])):
            with torch.no_grad():
                i_2_temp1=pre_acts[:,neuron_ind,:].unsqueeze(-1)
                i_2_temp2=pre_acts[:,neuron_ind,:].unsqueeze(-2)

                i_2=i_2_temp1+i_2_temp2
                i_2_nonint=i_2_temp1.abs()+i_2_temp2.abs()
                #set diagonals = 0
                i_2=i_2* ~torch.eye(i_2.shape[-1], dtype=bool).view(1, i_2.shape[-1], i_2.shape[-1])
                i_2_nonint=i_2_nonint* ~torch.eye(i_2.shape[-1], dtype=bool).view(1, i_2.shape[-1], i_2.shape[-1])

                i_2_metric=(1/(2*(i_2.shape[-1]-1)))*torch.sum(i_2.abs(),dim=0)
                i_2_nonint_metric=(1/(2*(i_2.shape[-1]-1)))*torch.sum(i_2_nonint,dim=0)

                i_2_tensor[neuron_ind,:]=i_2_metric
                i_2_nonint_tensor[neuron_ind,:]=i_2_nonint_metric

        return i_2_tensor,i_2_nonint_tensor


    # i_2_tensor,i_2_nonint_tensor=form_i2(pre_acts)
    # i_2_metric_tensor=torch.sum(i_2_tensor,dim=(1,2))
    # i_2_nonint_metric_tensor=torch.sum(i_2_nonint_tensor,dim=(1,2))

    # fig = make_subplots(rows=2, cols=2, subplot_titles=['i_1/i_N', 'i_2/i_N', 'i_2/i_1'], specs=[[{}, {}], [{"colspan": 2}, None]])
    # fig.add_trace(go.Scatter(x=np.arange(i_1.shape[0]),y=(i_1/i_inf).detach().numpy().flatten(),name='i_1',mode='markers'),row=1,col=1)
    # fig.add_trace(go.Scatter(x=np.arange(i_2_tensor.shape[0]),y=(i_2_metric_tensor/i_inf).detach().numpy().flatten(),name='i_2',mode='markers'),row=1,col=2)
    # fig.add_trace(go.Scatter(x=np.arange(i_2_tensor.shape[0]),y=(i_2_metric_tensor/i_1).detach().numpy().flatten(),name='i_2',mode='markers'),row=2,col=1)
    # #fig.add_trace(go.Scatter(x=np.arange(i_inf.shape[0]),y=(i_inf/i_inf).detach().numpy().flatten(),name='i_inf',mode='markers'),row=1,col=1)
    # fig.update_xaxes(title='MLP Neuron')
    # fig.update_yaxes(title='ratio to i_inf',row=1)
    # fig.update_yaxes(title='ratio i_2/i_1',row=1)
    # # Set the same y-axis range for both plots in first row
    # y_values = np.concatenate([
    #     (i_1/i_inf).detach().numpy().flatten(),
    #     (i_2_metric_tensor/i_inf).detach().numpy().flatten()
    # ])
    # y_min, y_max = np.min(y_values), np.max(y_values)
    # fig.update_yaxes(range=[y_min, y_max], row=1)
    # fig.show()
    # exit()
    #OK, that is pretty disappointing. Still, out of interest, let's have a look at the pairwise interaction measure
    #Let's just do the simple sum of the absolute values of the off-diagonal elements

    # pairwise_int=i_2_nonint_tensor-i_2_tensor
    # pairwise_int=einops.rearrange(pairwise_int,'neuron h1 h2 -> h1 h2 neuron')
    # print(f'pairwise_int shape {pairwise_int.shape}')
    # random_indices=torch.randperm(pairwise_int.shape[2])[:10]
    # imshow_fourier2(pairwise_int[:,:,random_indices])
    
    # fig=make_subplots(rows=1,cols=1)
    # fig.add_trace(go.Heatmap(z=torch.mean(pairwise_int,dim=0).detach().numpy(),colorscale='RdBu'),row=1,col=1)
    # fig.update_layout(title='Average pairwise interaction')
    # fig.show()
    
    
    
    

    
    

    

    
    #Mwahaha - now we have the tensor for all neurons, we should just be able to look at the feature interaction over neurons
    #First thing, let's look at the CDF of feature contributions to a given input

    #neuron_index=23
    #pre_acts_f=fft2d(pre_acts,fourier_basis,P)
    #pre_acts_f=rearrange_fourier_neel(pre_acts_f,P)
    #print(f'pre_acts_f shape {pre_acts_f.shape}')
    
    
    #imshow_fourier2(pre_acts_f[:,:,neuron_index,:])


    
    
    def plot_neuron_feature_cdf(pre_acts, neuron_index, num_samples=10):
        neuron_pre_acts=pre_acts[:,neuron_index,:].abs()
        sorted_neuron_pre_acts, _ = torch.sort(neuron_pre_acts, dim=1,descending=True)
        neuron_feat_cdf=torch.cumsum(sorted_neuron_pre_acts[:,:],dim=-1)
        #normalize
        neuron_feat_cdf=neuron_feat_cdf/neuron_feat_cdf[:,-1].unsqueeze(-1)
        # Sort along the second dimension (hidden dimension)
        
        print(f'neuron_feat_cdf shape {neuron_feat_cdf.shape}')
        fig=make_subplots(rows=1,cols=1)
        for input in range(num_samples):
            random_input=torch.randint(0,neuron_feat_cdf.shape[0],(1,))
            chosen_cdf=neuron_feat_cdf[random_input,:].squeeze()
            
            print(f'chosen_cdf shape {chosen_cdf.shape}')
            indices=np.arange(chosen_cdf.shape[0])
            print(f'indices shape {indices.shape}')
            
            fig.add_trace(go.Scatter(x=indices,y=chosen_cdf.detach().numpy(),mode='lines',name=f'input {random_input.item()}'),row=1,col=1)
        fig.update_layout(title='CDF of feature contributions to a given input at a single neuron')
        fig.update_xaxes(title='Ranked Feature index (sorted for each input)')
        fig.update_yaxes(title='Cumulative share of preactivation (*abs*)')
        return fig

   
    def pair_metric_sum(enc_acts:torch.Tensor,xcoder:AcausalCrosscoder,eps:float=1e-6):
        """
        The initial proposal for the pairwise metric, based on the mod sum.
        Issue here is it mixes interaction with polysemanticity.

        Expression on page 114 of "Project notes"
        """
        print(f'enc_acts shape {enc_acts.shape}')
        enc_acts=enc_acts+eps

        enc_acts_rearrange=einops.rearrange(enc_acts,'(batch sequence) hidden -> batch sequence hidden', sequence=3)
        enc_acts_last=enc_acts_rearrange[:,-1,:]
        #einops.rearrange(enc_acts,'(batch sequence) hidden -> batch sequence hidden',sequence=3)
        dec_ten=einops.einsum(W_in,(xcoder.W_dec_HMLD[:,0,1,:]),"d_mlp d_model, hidden d_model -> hidden d_mlp")
        dec_ten+=einops.einsum(W_in,(xcoder.b_dec_MLD[0,1,:]),"d_mlp d_model, d_model -> d_mlp")
        print(f'b_mlp_in shape {b_mlp_in.shape}')
        dec_ten+=b_mlp_in
        dec_ten=einops.rearrange(dec_ten,'hidden d_mlp -> d_mlp hidden')
        dec_ten=dec_ten.abs()+eps

        not_maxed=False
        max_norm=True
        if not_maxed:
            #should be shape (d_mlp,hidden,hidden)
            dec_pairs = dec_ten.unsqueeze(-1) / dec_ten.unsqueeze(-2)
        
            enc_sum_2=(enc_acts_last.unsqueeze(-1)/enc_acts_last.unsqueeze(-2)).abs().sum(dim=0)
            
            metric_tensor=dec_pairs*enc_sum_2.unsqueeze(0)
        elif max_norm:
            #should be shape (d_mlp,hidden,hidden)
            print(f'dec_ten shape {dec_ten.shape}')
            print(f'enc_acts_last shape {enc_acts_last.shape}')
            
            i_metric=dec_ten*enc_acts_last.sum(dim=0).unsqueeze(0)
            i_metric=i_metric/i_metric.max(dim=-1,keepdim=True).values
            
            
        return i_metric

    metric_tensor=pair_metric_sum(enc_acts,xc)
    cdf_metric_tensor=metric_tensor.sort(dim=-1,descending=True).values
    cdf_metric_tensor=torch.cumsum(cdf_metric_tensor,dim=-1)
    cdf_metric_tensor=cdf_metric_tensor/cdf_metric_tensor[:,-1].unsqueeze(-1)
    fig=make_subplots(rows=2,cols=2)

    # Calculate shared color range
    all_values = np.concatenate([
        metric_tensor.detach().numpy().flatten(),
        cdf_metric_tensor.detach().numpy().flatten()
    ])
    zmin, zmax = np.min(all_values), np.max(all_values)

    fig.add_trace(go.Heatmap(z=metric_tensor.detach().numpy(), colorscale='RdBu', zmin=zmin, zmax=zmax),row=1,col=1)
    fig.add_trace(go.Scatter(x=np.arange(metric_tensor.shape[1]),y=torch.mean(metric_tensor,dim=0).detach().numpy(),mode='lines',name='Mean over neurons'),row=1,col=2)
    fig.update_xaxes(title='Feature')
    fig.update_yaxes(title='Neuron',row=1,col=1)
    fig.update_yaxes(title='Mean over neurons',row=1,col=2)
    
    # Find indices where CDF first exceeds 0.9
    threshold_indices = torch.argmax((cdf_metric_tensor >= 0.9).float(), dim=1)
    
    
    # Create heatmap
    fig.add_trace(go.Heatmap(z=cdf_metric_tensor.detach().numpy(), colorscale='RdBu', zmin=zmin, zmax=zmax),row=2,col=1)
    
    # Add vertical lines at threshold points
    for neuron_idx, threshold_idx in enumerate(threshold_indices):
        fig.add_shape(
            type="line",
            x0=threshold_idx.item(), x1=threshold_idx.item(),
            y0=neuron_idx, y1=neuron_idx+1,
            line=dict(color="black", width=2),
            row=2, col=1
        )
    fig.add_trace(go.Scatter(x=np.arange(cdf_metric_tensor.shape[0]),y=cdf_metric_tensor[:,0].detach().numpy(),mode='lines',name='Dominant feature share'),row=2,col=2)
    fig.update_xaxes(title='Neuron',row=2,col=2)
    fig.update_yaxes(title='Dominant feature share of interaction metric',row=2,col=2)
    
    fig.update_layout(title='Pairwise metric tensor')
   
    
    fig.show()

    
    exit()

        
        
    def heatmap_metric_tensor(metric_tensor:torch.Tensor,random_samples:int=20):    
    
        #metric_tensor=pair_metric_sum(enc_acts,xc)
        metric_tensor=einops.rearrange(metric_tensor,'d_mlp h1 h2 -> h1 h2 d_mlp')
        
        
        random_indices = torch.randperm(metric_tensor.shape[2])[:random_samples]
        imshow_fourier2(metric_tensor[:,:,random_indices])
    
    #Let' try to implement the pairwise measure!


    #First, I want to be able to implement the decomposition

    heatmap_metric_tensor(pair_metric_sum(enc_acts,xc))

