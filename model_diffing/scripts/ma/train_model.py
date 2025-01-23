import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from models.ma_transformer import Transformer,TransformerConfig
from dataloader.ma_dataset import datacfg,gen_train_test,get_is_train_test
import copy
from datetime import datetime
from tqdm import tqdm

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import matplotlib.pyplot as plt
from IPython.display import clear_output

import plotly.colors as plc
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def cross_entropy_high_precision(logits, labels):
    # Shapes: batch x vocab, batch
    # Cast logits to float64 because log_softmax has a float32 underflow on overly
    # confident data and can only return multiples of 1.2e-7 (the smallest float x
    # such that 1+x is different from 1 in float32). This leads to loss spikes
    # and dodgy gradients
    logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1)
    loss = -torch.mean(prediction_logprobs)
    return loss

def get_accuracy(logits, labels):
    # Shapes: batch x vocab, batch
    predictions = torch.argmax(logits, dim=-1)
    correct = torch.eq(predictions, labels).float()
    accuracy = torch.mean(correct)
    return accuracy

def full_loss(model, data,labels):
    # Take the final position only
    logits = model(data)[:, -1]
    return cross_entropy_high_precision(logits, labels)

def accuracy(model,data,labels):

    # Take the final position only
    logits = model(data)[:, -1]
    return get_accuracy(logits, labels)

def train_loop(model,model_cfg,train_set,train_labels,test_set,test_labels,save=False):
    train_losses=[]
    train_accs=[]
    test_losses=[]
    test_accs=[]

    optimizer = model_cfg.optimizer(model.parameters(), lr=trans_cfg.lr, weight_decay=trans_cfg.weight_decay,betas=(0.9, 0.98))
    
    start_time=datetime.now()
    stripped_start_time=start_time.strftime("%Y-%m-%d_%H-%M-%S")

    first_train_loss=full_loss(model,train_set,train_labels)
    first_train_acc=accuracy(model,train_set,train_labels)
    first_test_loss=full_loss(model,test_set,test_labels)
    first_test_acc=accuracy(model,test_set,test_labels)

    train_losses.append(first_train_loss.item()),train_accs.append(first_train_acc.item()),test_losses.append(first_test_loss.item()),test_accs.append(first_test_acc.item())    
    
    save_dict={}
    save_dict[0] = {"model": copy.deepcopy(model.state_dict()),
                    "optimiser": copy.deepcopy(optimizer.state_dict()),
                    # 'scheduler': scheduler.state_dict(),
                    "train_acc": first_train_acc,
                    "test_acc": first_test_acc,
                    "train_loss": first_train_loss,
                    "test_loss": first_test_loss,
                    "epoch": 0,
                }

    if save:
        os.makedirs(save_dir,exist_ok=True)
        filename=f'{save_dir}/train_P_{model_cfg.P}_tf_{data_cfg.train_frac}_lr_{model_cfg.lr}_{stripped_start_time}.pt'
        
        torch.save(save_dict,filename)
    with tqdm(total=model_cfg.epochs, desc="Training") as pbar:
        for epoch in range(model_cfg.epochs):
            optimizer.zero_grad()
            train_loss=full_loss(model,train_set,train_labels)
            train_loss.backward()
            optimizer.step()
            train_acc=accuracy(model,train_set,train_labels)

            
            with torch.no_grad():
                test_loss=full_loss(model,test_set,test_labels)
                test_acc=accuracy(model,test_set,test_labels)
            
            train_losses.append(train_loss.item()),train_accs.append(train_acc.item()),test_losses.append(test_loss.item()),test_accs.append(test_acc.item())
            
            #update progress bar
            # Update progress bar
            pbar.set_postfix(acc=f"{100 * test_acc:.1f}%", loss=f"{train_loss.item():.4f}")
            pbar.update(1)  # Move progress by 1 epoch

            if epoch%trans_cfg.save_interval==0 or (epoch==model_cfg.epochs-1):
                save_dict[epoch+1]={
                    "model": copy.deepcopy(model.state_dict()),
                    "optimiser": copy.deepcopy(optimizer.state_dict()),
                    # 'scheduler': scheduler.state_dict(),
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "epoch": epoch+1,#because I save the first one
                }

                pbar.set_postfix(acc=f"{100*test_acc:.1f}"+"%")
                pbar.update(1) 

                if save:
                    torch.save(save_dict,filename)
                print(f'Epoch {epoch}:\n Train loss: {train_loss}\n Test loss: {test_loss}\n Train accuracy: {train_acc}\n Test accuracy: {test_acc}')
        
    return model,train_losses,train_accs,test_losses,test_accs

def plot_progress(train_losses,train_accs,test_losses,test_accs):
        """Plot the training progress."""
        clear_output(wait=True)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
    
        plt.plot(np.array(train_losses), label='Train Loss')
        plt.plot(np.array(test_losses), label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curve')
        
        plt.subplot(1, 2, 2)
        plt.plot(np.array(train_accs), label='Train Accuracy')
        plt.plot(np.array(test_accs), label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy Curve')
        plt.show()

def quick_end_plot(train_losses,test_losses,train_accs,test_accs):

    fig=make_subplots(rows=1,cols=2,subplot_titles=('Accuracy','Loss'))
    fig.add_trace(go.Scatter(y=test_losses,mode='lines',name='Train Accuracy',line=dict(color='red')),row=1,col=2)
    fig.add_trace(go.Scatter(y=train_losses,mode='lines',name='Train Loss',line=dict(color='blue')),row=1,col=2)
    fig.add_trace(go.Scatter(y=test_accs,mode='lines',name='Test Accuracy',line=dict(color='red')),row=1,col=1)
    fig.add_trace(go.Scatter(y=train_accs,mode='lines',name='Train Accuracy',line=dict(color='blue')),row=1,col=1)
    fig.update_yaxes(title_text='Accuracy',row=1,col=1)
    fig.update_yaxes(title_text='Loss',type='log',row=1,col=2)
    fig.update_xaxes(title_text='Epoch')
    fig.update_layout(title=f'LR: {trans_cfg.lr}, wd: {trans_cfg.weight_decay}')
    
    return fig


if __name__=="__main__":
    print('the main character')
    save_dir='/Users/dmitrymanning-coe/Documents/Research/Compact Proofs/code/toy_models2/data'
    #Initialize the model
    trans_cfg = TransformerConfig(
        P=23,
        lr=1e-3,
        weight_decay=1e-5,
        epochs=5000,
        save_interval=100,
    )
    model = Transformer(trans_cfg)
    print(f'Transformer model:\n {model}')
    print(f'Transformer cfg:\n {trans_cfg}')

    #Load in the data
    data_cfg=datacfg(
        P=23,
        num=1,
        data_seed=0,
        train_frac=0.8
    )

    train_set,test_set,train_labels,test_labels=gen_train_test(data_cfg)
    is_train,is_test=get_is_train_test(train_set,data_cfg)

    print(f'Train data shape:\n {np.array(train_set).shape}')
    print(f'Test data shape:\n {np.array(test_set).shape}')

    #sample model output

    model_output=model(train_set)
    train_loss=full_loss(model,train_set,train_labels)

    print(f'Train loss:\n {train_loss}')
    print(f'Train accuracy:\n {accuracy(model,train_set,train_labels)}')


    #train the model

    model,train_losses,train_accs,test_losses,test_accs=train_loop(model,trans_cfg,train_set,train_labels,test_set,test_labels,save=False)

    quick_end_plot(train_losses,test_losses,train_accs,test_accs).show()

    
    






    
    
