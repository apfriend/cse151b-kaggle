import torch
import pandas as pd
import numpy as np
from torch import nn
from tqdm import tqdm
from IPython.display import display

class simple_model(nn.Module):
    def __init__(self, device):
        super(simple_model, self).__init__()
        
        self.hidden_dim=2048
        self.num_layers=3
        self.device=device

        #simple single layer linear model
        self.linear=nn.Linear(
            in_features=240*19,
            out_features=240*30
        )
        
    def forward(self, x):        
        x=self.linear(x)
        return x
        
    def forward_test(self, x, num_steps=30):
        res=[]
        # h=torch.zeros((self.num_layers, len(x), self.hidden_dim)).to(self.device)
        # c=torch.zeros((self.num_layers, len(x), self.hidden_dim)).to(self.device)
        for step in range(num_steps):
            # x, (h,c)=self.lstm(x, (h,c))
            x=x.linear(x)
            x=x[:,-1:]
            x=x.transpose(1,2)
            x=self.linear(x)
            x=x.transpose(1,2)
            res.append(x)
        res=torch.cat(res,1)
        return res

class multilayer_model(nn.Module):
    def __init__(self, device):
        super(multilayer_model, self).__init__()
        
        self.hidden_dim=8192
        self.num_layers=3
        self.device=device
                
        self.linear=nn.Sequential(
            nn.Linear(
                in_features=240*19, 
                out_features=self.hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=self.hidden_dim,
                out_features=240*30
            )
        )
        
    def forward(self, x):        
        x=self.linear(x)
        return x
        
    def forward_test(self, x, num_steps=30):
        res=[]
        # h=torch.zeros((self.num_layers, len(x), self.hidden_dim)).to(self.device)
        # c=torch.zeros((self.num_layers, len(x), self.hidden_dim)).to(self.device)
        for step in range(num_steps):
            # x, (h,c)=self.lstm(x, (h,c))
            x=x.linear(x)
            x=x[:,-1:]
            x=x.transpose(1,2)
            x=self.linear(x)
            x=x.transpose(1,2)
            res.append(x)
        res=torch.cat(res,1)
        return res

def train(model, data_loader, n_epochs, filename, ema_weight, device='cuda', verbose=False):
    '''
    Train LSTM model
    ----------
    Parameters
    ----------
        lmodel - torch nerual network linear model
            model to train
        data_loader - torch DataLoader class 
            training data for model
        n_epochs - int
            number of epochs to train
        filename - string
            filepath to save training data to
        ema_weight - float
            float between (0.0,1.0) for the exponential moving average weight
        device - string, default 'cuda'
            choose to run on gpu ('cuda') or cpu ('CPU')
        verbose - boolean, default False
            If true print training progress every 10 training iterations
    -------
    Returns
    -------
         Trained LSTM model
    '''
    
    model=model(device).to(device)
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3)

    loss_ema=-1

    data=[]

    for epoch in range(n_epochs):
        for i_batch, sample_batch in enumerate(tqdm(data_loader, desc='Epoch %i/%i'%(epoch+1,n_epochs), disable=verbose)):
            '''
            TODO:
            Deep learning model training routine
            '''
            inp, out=sample_batch
            inp, out=inp.to(device), out.to(device)

            # input: batch size x 60 x 49 x 4
            # transpose: batch size x 49 x 240
            mixed=torch.cat([inp, out], 2).transpose(1,2).reshape((-1,49,240))#.float()

            y_pred=model(inp.reshape((len(inp),-1))).reshape((-1,60,30,4))

            loss=(torch.mean((y_pred-out)**2))**0.5
            optimizer.zero_grad() # set gradient to zero
            loss.backward() # backwards propogation
            optimizer.step() #forward step

            if loss_ema<0:
                loss_ema=loss
            loss_ema=loss_ema*ema_weight+loss*(1-ema_weight)

            if verbose and i_batch%100==0:
                loss_str='losss %i %i %f %f'%(epoch,i_batch,loss_ema.item(),loss.item())
                print(loss_str)
                # print([y_pred.detach().cpu().numpy(), out.detach().cpu().numpy()])
                
            pos_idx=torch.tensor([0,1]).to(device)
            positions=torch.index_select(y_pred.to(device), 3, pos_idx)
            out=torch.index_select(out.to(device), 3, pos_idx)

        if i_batch%100==0:
            data.append([epoch, i_batch, loss_ema.item(), loss.item(), positions.detach().cpu().numpy(), out.detach().cpu().numpy()])
    
    columns=["epoch","iteration","loss_ema","loss"]
    df=pd.DataFrame(dict(zip(columns, np.array(data).T)))
    display(df)
    df.to_csv(filename, index=False)
    return model