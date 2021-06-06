from matplotlib.pyplot import plot
import torch
# import multiprocessing
import pandas as pd
import numpy as np
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from time import time
from IPython.display import display
#####LOCAL SCRIPTS#####
import utility
import run_model
# import load_data
# from ArgoverseDataset import ArgoverseDataset
# from load_data import DataLoader, collate_train
# from util_functions import get_avg_loss, get_avg_epoch_loss, plot_loss, plot_avg_epoch_loss

class simple_model(nn.Module):
    def __init__(self, device, dropout):
        super(simple_model, self).__init__()
        
        self.hidden_dim=2048
        self.num_layers=1
        self.device=device
        self.dropout=dropout

        if self.dropout:
            self.linear=nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(
                    in_features=240*19,
                    out_features=240*30
                )
            )
        else:
            self.linear=nn.Linear(
                in_features=240*19,
                out_features=240*30
            )

    def forward(self, x):        
        x=self.linear(x)
        return x
        
    def forward_test(self, x, num_steps=30):
        res=[]
        for step in range(num_steps):
            x=x.linear(x)
            x=x[:,-1:]
            x=x.transpose(1,2)
            x=self.linear(x)
            x=x.transpose(1,2)
            res.append(x)
        res=torch.cat(res,1)
        return res

class multilayer_model2(nn.Module):
    def __init__(self, device, dropout):
        super(multilayer_model2, self).__init__()
        
        self.input_dim=240*19
        self.output_dim=240*30
        self.hidden_dim1=2048
        self.hidden_dim2=4096
        self.num_layers=4
        self.device=device

        if dropout:
            self.linear=nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(
                    in_features=self.input_dim,
                    out_features=self.hidden_dim1
                ),
                nn.ReLU(),
                nn.Linear(
                    in_features=self.hidden_dim1,
                    out_features=self.hidden_dim2
                ),
                nn.ReLU(),
                nn.Linear(
                    in_features=self.hidden_dim2,
                    out_features=self.hidden_dim1
                ),
                nn.Sigmoid(),
                nn.Linear(
                    in_features=self.hidden_dim1,
                    out_features=self.output_dim
                )
            )
        else:
            self.linear=nn.Sequential(
                nn.Linear(
                    in_features=self.input_dim,
                    out_features=self.hidden_dim1
                ),
                nn.ReLU(),
                nn.Linear(
                    in_features=self.hidden_dim1,
                    out_features=self.hidden_dim2
                ),
                nn.ReLU(),
                nn.Linear(
                    in_features=self.hidden_dim2,
                    out_feature=self.hidden_dim1
                ),
                nn.Sigmoid(),
                nn.Linear(
                    in_features=self.hidden_dim1,
                    out_features=self.output_dim
                )
            )

    def forward(self, x):        
        x=self.linear(x)
        return x
        
    def forward_test(self, x, num_steps=30):
        res=[]
        for step in range(num_steps):
            x=x.linear(x)
            x=x[:,-1:]
            x=x.transpose(1,2)
            x=self.linear(x)
            x=x.transpose(1,2)
            res.append(x)
        res=torch.cat(res,1)
        return res

class multilayer_model(nn.Module):
    def __init__(self, device, dropout):
        super(multilayer_model, self).__init__()
        
        # self.hidden_dim=8096
        self.hidden_dim=1024
        self.num_layers=3
        self.device=device
                
        self.linear=nn.Sequential(
            nn.Dropout(p=dropout),
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
        for step in range(num_steps):
            x=x.linear(x)
            x=x[:,-1:]
            x=x.transpose(1,2)
            x=self.linear(x)
            x=x.transpose(1,2)
            res.append(x)
        res=torch.cat(res,1)
        return res

def train(model, data_loader, test_data, n_epochs, filename, ema_weight, dropout, batch_sz, 
    device='cuda:0', verbose=False, normalize=False):
    '''
    Train linear model
    ----------
    Parameters
    ----------
        model - torch nerual network linear model
            model to train
        data_loader - torch DataLoader class 
            training data for model
        test_data - list of torch.FloatTensor objects
            test data to get test loss of model. Same shape as data from <data_loader>
        n_epochs - int
            number of epochs to train
        filename - string
            filepath to save training data to
        ema_weight - float
            float between (0.0,1.0) for the exponential moving average weight
        batch_sz - int
            size of minibatches
        device - string, default 'cuda:0'
            choose to run on gpu ('cuda') or cpu ('CPU')
        verbose - boolean, default False
            If true print training progress every 10 training iterations
        normalize - boolean, default False
            If true normalize training data.
    -------
    Returns
    -------
        (model, df)
        1. Trained linear model
        2. Pandas Dataframe of loss for each iteration
    '''    
    if dropout:
        model=model(device, dropout).to(device)
    else:
        model=model(dropout, 0).to(device)

    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion=utility.rmse
    
    num_elems=60
    batch_norm=nn.BatchNorm2d(num_elems).to(device)

    train_loss_ema=-1
    test_loss_ema=-1

    # test_data=test_data.to(device)
    test_inp, test_out=test_data
    test_inp, test_out=test_inp.to(device), test_out.to(device)
    if normalize:
        test_inp=batch_norm(test_inp)
    
    data_columns=["epoch","iteration","train_loss_ema","train_loss", "test_loss_ema","test_loss"]    
    num_batches=len(data_loader)
    num_rows=num_batches*n_epochs
    num_cols=len(data_columns)
    data_shape=(num_rows, num_cols)
    data=np.empty(data_shape)

    idx=0
    start_time=time()
    for epoch in range(1, n_epochs+1):
        for i_batch, sample_batch in enumerate(tqdm(data_loader, desc='Epoch %i/%i'%(epoch,n_epochs), 
                                                    disable=verbose, leave=False)):
            inp, out=sample_batch
            inp, out=inp.to(device), out.to(device)
            if normalize:
                inp=batch_norm(inp)

            # input: batch size x 60 x 49 x 4
            # transpose: batch size x 49 x 240
            # mixed=torch.cat([inp, out], 2).transpose(1,2).reshape((-1,49,240))#.float()
            y_pred=run_model.model_predict(model, inp)

            # train_loss=(torch.mean((y_pred-out)**2))**0.5
            train_loss=criterion(y_pred, out)
            optimizer.zero_grad() # set gradient to zero
            train_loss.backward() # backwards propogation
            optimizer.step() #forward step

            # get test loss
            with torch.no_grad():        
                test_pred=run_model.model_predict(model, test_inp)
                test_loss=criterion(test_pred, test_out)

            if train_loss_ema<0:
                train_loss_ema=train_loss
            train_loss_ema=train_loss_ema*ema_weight+train_loss*(1-ema_weight)
            
            if test_loss_ema<0:
                test_loss_ema=test_loss
            test_loss_ema=test_loss_ema*ema_weight+test_loss*(1-ema_weight)

            if verbose and i_batch%100==0:
                loss_str='losss %i %i %f %f'%(epoch,i_batch,train_loss_ema.item(),train_loss.item(),test_loss_ema.item(),test_loss.item())
                print(loss_str)
                
            pos_idx=torch.tensor([0,1]).to(device)
            positions=torch.index_select(y_pred.to(device), 3, pos_idx)
            out=torch.index_select(out.to(device), 3, pos_idx)

            data[idx]=np.array([epoch,i_batch,train_loss_ema.item(),train_loss.item(),test_loss_ema.item(),test_loss.item()])
            idx+=1

        if not verbose:
            current_row=idx-1
            print_data=(epoch, data[current_row,2], data[current_row,3], data[current_row,4], data[current_row,5])
            print('Epoch: %s, Train Loss: %s, Train EMA Loss: %s Test Loss: %s, Test EMA Loss: %s'%print_data, flush=True)
    
    df=pd.DataFrame(
        data=data,
        columns=data_columns
    )

    df.to_csv(filename, index=False)   
    
    seconds=int(time()-start_time)
    hours, minutes, seconds=utility.convert_seconds(seconds)
    print("Completed training in %s hours,%s minutes, %s seconds"%(hours, minutes, seconds))
    display(df)
    
    return model, df

