import torch
from model_diffing.models.ma_transformer import Transformer,TransformerConfig
from typing import Any,Union,Dict,Any


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
First want to get an activations tensor from the model.
Would be helpful to have to keep it in the structure of the xcoder.
"""


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

def get_activations(model:Transformer,P:int)->Dict[str, Any]:
    all_data = torch.tensor([(i, j, P) for i in range(P) for j in range(P)]).to(device)
    labels = torch.tensor([(i+j)%P for i, j, _ in all_data]).to(device)
    cache = {}
    #model.remove_all_hooks()
    model.cache_all(cache)
    model(all_data)
    model.remove_all_hooks()
    return cache
