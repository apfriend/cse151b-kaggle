import os
import re
import pandas as pd
import shutil
import random
import warnings
from glob import glob
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import random
import datetime as dt
from tqdm import tqdm
import seaborn as sns
from IPython.display import display
#####LOCAL SCRIPTS#####
import run_model
import load_data
import models.physics as physics
import models.linear as linear

def visualize_trajectory(inp, target, predicted, save_path=None):
    '''
    Visualize the trajectory for a of the training data, actual outputs, and predicted outputs
    ----------
    Parameters
    ----------
        inp - torch.FloatTensor object
            input data used for prediction
        target - torch.FloatTensor object
            Actual output position data
        predicted - torch.FloatTensor object
            Predicted output position data
        save_path - str. Default None
            If not none save figure to path argument
    '''
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            category=UserWarning
        )
        num_samples=5
        agent_ids=[]
        while len(agent_ids)<num_samples:
            random_int=random.randint(0,59)
            if random_int not in agent_ids:
                agent_ids.append(random_int)

        batch_sz = inp.size(0)
        agent_sz = inp.size(1)
        
        num_plots=3
        fig, axs = plt.subplots(
            nrows=num_samples,
            ncols=num_plots,
            figsize=(15, 20), 
            facecolor='w', 
            edgecolor='k'
        )
        fig.subplots_adjust(hspace = .5, wspace=.001)

        for idx, agent_id in enumerate(agent_ids):
            # first two feature dimensions are (x,y) positions
            axs[idx,0].scatter(
                x=inp[idx, agent_id,:,0].detach().cpu().numpy(), 
                y=inp[idx, agent_id,:,1].detach().cpu().numpy()
            )
            axs[idx,1].scatter(
                x=target[idx, agent_id,:,0].detach().cpu().numpy(), 
                y=target[idx, agent_id,:,1].detach().cpu().numpy()
            )
            axs[idx,2].scatter(
                x=predicted[idx, agent_id,:,0].detach().cpu().numpy(), 
                y=predicted[idx, agent_id,:,1].detach().cpu().numpy()
            )
        
        axs[0,0].set_title('Input Positions')
        axs[0,1].set_title('Actual Output Positions')
        axs[0,2].set_title('Predicted Output Positions')

        if save_path:
            fig.savefig(save_path)    
        fig.show()

def get_avg_epoch_loss(df):
    '''
    Get the average loss across each epoch
    '''
    return df.groupby(["epoch"]).mean()

def get_avg_loss(df, num_iter):
    '''
    Get the average loss across every <num_iter> mini_batch iterations
    '''
    avg_data={
        "iteration":[],
        "train_loss_ema":[],
        "train_loss":[],
        "test_loss_ema":[],
        "test_loss":[]
    }
    for i in range(0,df.shape[0], num_iter):
        if not i>=df.shape[0]:
            avg_df=df.iloc[i:i+num_iter,:]
        else:
            avg_df=df.iloc[i:df.shape[0]-1,:]
        avg_data["iteration"].append(avg_df.iteration.mean())
        avg_data["train_loss_ema"].append(avg_df.train_loss_ema.mean())
        avg_data["train_loss"].append(avg_df.train_loss.mean())
        avg_data["test_loss_ema"].append(avg_df.test_loss_ema.mean())
        avg_data["test_loss"].append(avg_df.test_loss.mean())
    return pd.DataFrame(avg_data)

def plot_avg_epoch_loss(df, save_path=None):
    '''
    Scatter plot average loss_ema and loss for each epoch
    '''
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            category=UserWarning
        )

        font = {
            'family' : 'DejaVu Sans',
            'weight' : 'bold',
            'size'   : 14
        }

        plt.rc('font', **font)

        fig, axs=plt.subplots(nrows=2,ncols=2, figsize=(10,5))

        plt.subplots_adjust(wspace=0.3)

        fig.suptitle("Average Training Loss for each Epoch")
        fig.patch.set_facecolor("white")

        axs[0,0].scatter(
            x=df.index,
            y=df.train_loss_ema
        )
        axs[0,1].scatter(
            x=df.index,
            y=df.train_loss
        )
        axs[1,0].scatter(
            x=df.index,
            y=df.test_loss_ema
        )
        axs[1,1].scatter(
            x=df.index,
            y=df.test_loss
        )

        axs[0,0].set_ylabel("Train EMA Loss")
        axs[0,0].set_xlabel("Epoch")
        axs[0,1].set_ylabel("Train Loss")
        axs[0,1].set_xlabel("Epoch")
        axs[1,0].set_ylabel("Train EMA Loss")
        axs[1,0].set_xlabel("Epoch")
        axs[1,1].set_ylabel("Train Loss")
        axs[1,1].set_xlabel("Epoch")


        if save_path:
            fig.savefig(save_path)
        
        fig.show()

def plot_loss(df, save_path=None):
    '''
    Scatter plot df.loss_ema and df.loss across minibatch iterations
    '''
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            category=UserWarning
        )

        font = {
            'family' : 'DejaVu Sans',
            'weight' : 'bold',
            'size'   : 14
        }

        plt.rc('font', **font)

        fig, axs=plt.subplots(2,2, figsize=(10,7))

        plt.subplots_adjust(wspace=0.3, hspace=0.5)

        fig.suptitle("Average Loss over minibatch iterations")
        fig.patch.set_facecolor("white")

        axs[0,0].scatter(
            x=df.iteration,
            y=df.train_loss_ema
        )
        axs[0,1].scatter(
            x=df.iteration,
            y=df.train_loss
        )
        axs[1,0].scatter(
            x=df.iteration,
            y=df.test_loss_ema
        )
        axs[1,1].scatter(
            x=df.iteration,
            y=df.test_loss
        )

        axs[0,0].set_ylabel("Train EMA Loss")
        axs[0,0].set_xlabel("Minibatch iterations")
        axs[0,1].set_ylabel("Train Loss")
        axs[0,1].set_xlabel("Minibatch iterations")
        axs[1,0].set_ylabel("Test EMA Loss")
        axs[1,0].set_xlabel("Minibatch iterations")
        axs[1,1].set_ylabel("Test Loss")
        axs[1,1].set_xlabel("Minibatch iterations")

        if save_path:
            fig.savefig(save_path)
        
        fig.show()

def copy_file(args):
    '''
    Helper function to copy files from <src> to <dst>
    ----------
    Parameters
    ----------
        args - tuple 
            Tuple of strings
            
            First element <args[0]> contains the filepath of the file to copy

            Second element <args[1]> contains the directory to copy file top
    '''
    src=args[0]
    dst=args[1]
    fn=os.path.basename(src)
    dst=os.path.join(dst, fn)
    shutil.copyfile(src, dst)
    print(dst)

def sample_files(src, dst, sample_size):
    '''
    Sample files from <src> and copies them to <dst>
    ----------
    Parameters
    ----------
        src - str
            file path to directory storing files    
        dst - str
            file path to directory to store sampled files
        sample_size - int
            Number of files to sample
    '''
    files=glob(os.path.join(src,'*'))
    sample_files=random.sample(files, sample_size)
    os.makedirs(dst, exist_ok=True)
    
    for fp in sample_files:
        fn=os.path.basename(fp)
        file_dst=os.path.join(dst, fn)
        shutil.copyfile(fp, file_dst)

def dataloader_to_csv(data_loader, dst, in_fn, out_fn):
    '''
    Function to write dataloader object to csv file
    ----------
    Parameters
    ----------
        data_loader - PyTorch DataLoader object
            DataLoader object to parse to get data to save to csv
        dst - str
            filepath of directory to save csv file to
        in_fn - str
            filename to save in data csv file as
        out_fn -str
            filename to save out data csv file as
    '''
    data_in={}
    data_out={}

    cols=['x_pos','y_pos','x_vel','y_vel']
    for col in cols:
        data_in[col]=[]
        data_out[col]=[]

    in_fp=os.path.join(dst, in_fn)
    out_fp=os.path.join(dst, out_fn)

    pd.DataFrame(data_in).to_csv(in_fp, index=False)
    pd.DataFrame(data_out).to_csv(out_fp, index=False)

    for batch in tqdm(data_loader):
        
        data_in={}
        data_out={}
        for col in cols:
            data_in[col]=[]
            data_out[col]=[]
        
        in_data=batch[0].detach().numpy()
        out_data=batch[1].detach().numpy()
        
        for scene in in_data:
            for car in scene:
                for time_data in car:
                    data_in['x_pos'].append(time_data[0])
                    data_in['y_pos'].append(time_data[1])
                    data_in['x_vel'].append(time_data[2])
                    data_in['y_vel'].append(time_data[3])
        
        for scene in out_data:
            for car in scene:
                for time_data in car:
                    data_out['x_pos'].append(time_data[0])
                    data_out['y_pos'].append(time_data[1])
                    data_out['x_vel'].append(time_data[2])
                    data_out['y_vel'].append(time_data[3])
                
        pd.DataFrame(data_in).to_csv(in_fp, mode='a', header=False, index=False)
        pd.DataFrame(data_out).to_csv(out_fp, mode='a', header=False, index=False)

def get_agent_idx(agent_id, track_ids):
    '''
    Get index of track_id that is for agent_id
    ----------
    Parameters
    ----------
        agent_id - str
            string car id to track
        track_ids - numpy array
            a (60,30,1) shaped np array containing agent_ids of all cars tracked
    -------
    Returns
    -------
        The index of <track_id> that contains <agent_id>
    '''
    for i in range(track_ids.shape[0]):
        if agent_id in track_ids[i].reshape(1,-1)[0]:
            return i

def get_optimizer(optimizer, learning_rate):
    '''
    Get optimizer object
    ----------
    Parameters
    ----------
        optimizer - str
            name of type of optimizer to return
        learning_rate - float
            optimizer learning rate
    -------
    Returns
    -------
        return torch.optim optimizer object
    '''
    return None

def rmse(predicted,target):
    '''
    Root Mean Squared error
    ----------
    Parameters
    ----------
        predicted - torch.FloatTensor
            Floating point tensor of predicted values
        target - torch.FloatTensor
            Floating point tensor of actual values to be approximated in training
    -------
    Returns
    -------
        float of root mean square error of <y_pred> and <y_actual>
    '''
    mse=nn.MSELoss()
    return torch.sqrt(mse(predicted, target))

def convert_seconds(seconds):
    '''
    Convert seconds to hours, minutes, seconds
    ----------
    Parameters
    ----------
        seconds - int
            Total number of seconds to convert to hours, remaining minutes, and remaining seconds
    -------
    Returns
    -------
        hours, minutes, seconds
    '''
    hours=seconds//3600
    minutes=(seconds%3600)//60
    seconds=seconds%60
    return hours, minutes, seconds  

def get_model_data(fn):
    '''Get model name, dropout, batchsize and number of epochs from filename'''
    fn_list=fn.split('-')
    model_name=' '.join(fn_list[1:3])
    dropout=fn_list[3]
    m=re.search(r'\d+', dropout)

    dropout='0.'+m.group(0)
    dropout=float(dropout)
    batch_size=fn_list[4]
    m=re.search(r'\d+', batch_size)

    batch_size=int(m.group(0))
    epochs=fn_list[5]
    m=re.search(r'\d+', epochs)

    epochs=int(m.group(0))
    return model_name, dropout, batch_size, epochs

def compare_rmse(src, data_path, device='cuda:0'):
    '''
    Compare the Root Mean Square Error loss across models
    ----------
    Parameters
    ----------
        src - str
            path to directory where model state dictionaries are stored
        data_path - str
            path to directory where test data is stored
        device - str. Default: `cuda:0`
            Name of device to load model and data into.
    -------
    Returns
    -------
        Pandas DataFrame object of model performance
    '''
    model_fns=['PHYSICS']+os.listdir(src)
    # model_fns.append('PHYSICS')

    col_names=['model_name', 'rmse', 'dropout', 'batch_size', 'num_epochs']
    model_data=[[None]*5]*len(model_fns)

    find_model={
        'simple linear':linear.simple_model,
        'multilayer linear': linear.multilayer_model,
        'multilayer linear2':linear.multilayer_model2
    }

    num_elems=60
    batch_norm=nn.BatchNorm2d(num_elems).to(device)

    position_idxs=torch.tensor([0,1]).to(device)
    iteration=0
    for model_fn in tqdm(model_fns):
        if model_fn=='PHYSICS':
            model_name='Average Velocity Model'
            dropout=None
            batch_size=128
            num_epochs=None
            
            model=physics.avg_velocity_model
            is_physics=True
        else:
            model_name, dropout, batch_size, num_epochs=get_model_data(model_fn)
            model_src=os.path.join(src, model_fn)
            model_obj=find_model[model_name]
            model_obj=model_obj(device, dropout)
            model=load_data.load_model(
                src=model_src,
                model=model_obj
            )
            is_physics=False
            
        inp, target=load_data.get_test_data(data_path, batch_size)
        inp, target=inp.to(device), target.to(device)    
        if not is_physics:
            inp=batch_norm(inp)

        pred=run_model.model_predict(
            model=model,
            data=inp,
            physics_model=is_physics
        )
        target_position=torch.index_select(target, 3, position_idxs)
        pred_position=torch.index_select(pred, 3, position_idxs)
        loss=rmse(pred_position, target_position).detach().cpu().numpy()
            
        data=[model_name, loss, dropout, batch_size, num_epochs]
        model_data[iteration]=data
        iteration+=1            
            
    model_df=pd.DataFrame(
        data=model_data,
        columns=col_names
    )

    model_df.rmse=model_df.rmse.astype(np.float32)
    
    return model_df

def visualize_model_loss(df, model, dst=None):
    '''
    Visualize loss of model for different dropout, batch_size, or training epochs
    ----------
    Parameters
    ----------
        df - Pandas DataFrame
            Dataframe of models with their respective test losses, dropout rates, batch sizes, and training epochs
        model - str
            Name of model to visualize
        dst - str
            Path to save to
    '''
    df=df.loc[df.model_name==model].copy()
    batch_df=df.drop(columns=['dropout']).drop_duplicates(subset=['batch_size','num_epochs'])
    dropout_df=df.drop(columns=['batch_size']).drop_duplicates(subset=['dropout','num_epochs'])

    fig, (ax1, ax2)=plt.subplots(
        nrows=1,
        ncols=2, 
        figsize=(16,8),
        sharex=True,
        sharey=True
    )
    sns.lineplot(
        data=batch_df,
        x='num_epochs',
        y='rmse',
        hue='batch_size',
        ax=ax1
    )

    sns.lineplot(
        data=dropout_df,
        x='num_epochs',
        y='rmse',
        hue='dropout',
        ax=ax2
    )

    fig.suptitle('RMSE By Epochs, Batch Size, and Dropout', fontsize=24)
    ax1.set_ylabel('RMSE',fontsize=16)
    ax1.set_xlabel('Epochs',fontsize=16)
    ax2.set_xlabel('Epochs',fontsize=16)

    if dst:
        fig.savefig(dst)
    # sns.relplot(
    #     data=df,
    #     x='num_epochs',
    #     y='rmse',
    #     hue='batch_size',
    #     style='dropout',
    #     kind='line'
    # )