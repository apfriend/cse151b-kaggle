import torch
import pandas as pd
import numpy as np
from torch import nn
from tqdm import tqdm
from time import time
from IPython.display import display
#####LOCAL SCRIPTS#####
import utility

class LSTM_model(nn.Module):
    '''
    LSTM model
    '''
    def __init__(self, device):
        super(LSTM_model, self).__init__()
        # the LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim
        self.hidden_dim=2048
        self.num_layers=3
        self.device=device
        self.lstm=nn.LSTM(
            input_size=240,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True            
        )        
        # The linear layer that maps from hidden state space to tag space
        self.linear=nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=240,
            kernel_size=1
        )
        
    def forward(self, x):
        #batch_size x timesteps x 240
        x,_=self.lstm(x)
        # finally hidden layer batch_size x timesteps x hidden_dim
        x=x.transpose(1,2)
        # batch_size x hidden_dim x timesteps
        x=self.linear(x)
        x=x.transpose(1,2)        
        return x
        
    def forward_test(self, x, num_steps=30):
        res=[]
        h=torch.zeros((self.num_layers, len(x), self.hidden_dim)).to(self.device)
        c=torch.zeros((self.num_layers, len(x), self.hidden_dim)).to(self.device)
        for step in range(num_steps):
            x, (h,c)=self.lstm(x, (h,c))
            x=x[:,-1:]
            x=x.transpose(1,2)
            x=self.linear(x)
            x=x.transpose(1,2)
            res.append(x)
        res=torch.cat(res,1)
        return res

class LSTM_model2(nn.Module):
    '''
    LSTM model
    '''
    def __init__(self, device, dropout):
        super(LSTM_model, self).__init__()
        # the LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim
        self.hidden_dim=2048
        self.num_layers=3
        self.device=device
        self.dropout=dropout
        self.lstm=nn.LSTM(
            input_size=240,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True            
        )        
        # The linear layer that maps from hidden state space to tag space
        self.linear=nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=240,
            kernel_size=1
        )
        
    def forward(self, x):
        #batch_size x timesteps x 240
        x,_=self.lstm(x)
        # finally hidden layer batch_size x timesteps x hidden_dim
        x=x.transpose(1,2)
        # batch_size x hidden_dim x timesteps
        x=self.linear(x)
        x=x.transpose(1,2)        
        return x
        
    def forward_test(self, x, num_steps=30):
        res=[]
        h=torch.zeros((self.num_layers, len(x), self.hidden_dim)).to(self.device)
        c=torch.zeros((self.num_layers, len(x), self.hidden_dim)).to(self.device)
        for step in range(num_steps):
            x, (h,c)=self.lstm(x, (h,c))
            x=x[:,-1:]
            x=x.transpose(1,2)
            x=self.linear(x)
            x=x.transpose(1,2)
            res.append(x)
        res=torch.cat(res,1)
        return res

def train(model, data_loader, n_epochs, filename, ema_weight, device='cuda:0', verbose=False):
    '''
    Train LSTM model
    ----------
    Parameters
    ----------
        model - torch nerual network model
            model to train
        data_loader - torch DataLoader class 
            training data for model
        n_epochs - int
            number of epochs to train
        filename - string
            filepath to save training data to
        ema_weight - float
            float between (0.0,1.0) for the exponential moving average weight
        device - string, default 'cuda:0'
            choose to run on single gpu ('cuda:0') or cpu ('CPU')
        verbose - boolean, default False
            If true print training progress every 10 training iterations
    -------
    Returns
    -------
        Trained LSTM model
    '''    
    model=model.to(device)
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3)
    
    loss_ema=-1
    loss_ema2=-1

    data_columns=["epoch","iteration","loss_ema","loss","loss_ema2","loss2"]    
    num_batches=len(data_loader)
    num_rows=num_batches*n_epochs
    num_cols=len(data_columns)
    data_shape=(num_rows, num_cols)
    data=np.empty(data_shape)

    idx=0
    start_time=time()
    for epoch in range(n_epochs):
        for i_batch, sample_batch in enumerate(tqdm(data_loader, desc='Epoch %i/%i'%(epoch,n_epochs), 
                                                    disable=verbose, leave=False)):
            inp, out=sample_batch
            inp, out=inp.to(device), out.to(device)
            # input: batch size x 60 x 49 x 4
            # transpose: batch size x 49 x 240
            mixed=torch.cat([inp, out], 2).transpose(1,2).reshape((-1,49,240))#.float()

            y_pred=model(mixed[:,:-1])[:,-30:]
            y_pred=y_pred.reshape((-1,30,60,4)).transpose(1,2)

            loss=(torch.mean((y_pred-out)**2))**0.5
            optimizer.zero_grad() # set gradient to zero
            loss.backward() # backwards propogation
            optimizer.step() #forward step

            if loss_ema<0:
                loss_ema=loss
            loss_ema=loss_ema*ema_weight+loss*(1-ema_weight)

            with torch.no_grad():
                y_pred2=model.forward_test(inp.transpose(1,2).reshape((-1,19,240)))
                y_pred2=y_pred2.reshape((-1,30,60,4)).transpose(1,2)
                loss2=torch.mean((y_pred2-out)**2)**0.5
                if loss_ema2<0:
                    loss_ema2=loss2
                loss_ema2=loss_ema2*ema_weight+loss2*(1-ema_weight)

            data[idx]=np.array([epoch, i_batch, loss_ema.item(), loss.item(), loss_ema2.item(), loss2.item()])
            idx+=1      

            if verbose and i_batch%10==0:
                loss_str='Loss full %i %i %f %f'%(epoch,i_batch,loss_ema.item(),loss.item())
                loss2_str='EMA Loss full %i %i %f %f'%(epoch,i_batch,loss_ema2.item(),loss2.item())
                print(loss_str)
                print(loss2_str)
            
        if not verbose:
            current_row=idx-1
            print('Epoch: %s, Loss: %s, EMA Loss: %s'%(epoch, data[current_row,-1], data[current_row,-2]))
            # data.append([epoch, i_batch, loss_ema.item(), loss.item(), loss_ema2.item(), loss2.item()])
    
    # columns=["epoch","iteration","loss_ema","loss", "loss_ema2", "loss2"]
    # df=pd.DataFrame(dict(zip(columns, np.array(data).T)))
    # display(df)
    # df.to_csv(filename, index=False)
    # return model
    df=pd.DataFrame(
        data=data,
        columns=data_columns
    )

    df.to_csv(filename, index=False)   
    
    seconds=int(time()-start_time)
    hours, minutes, seconds=utility.convert_seconds(seconds)
    print("Completed training in %s hours,%s minutes, %s seconds"%(hours, minutes, seconds))
    display(df)
    
    return model