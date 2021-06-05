import os
import sys
import pandas as pd
import shutil
import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from glob import glob
import matplotlib.pyplot as plt
import random
import multiprocessing
from tqdm import tqdm
import seaborn as sns

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
        "loss_ema":[],
        "loss":[]
    }
    for i in range(0,df.shape[0], num_iter):
        if not i>=df.shape[0]:
            avg_df=df.iloc[i:i+num_iter,:]
        else:
            avg_df=df.iloc[i:df.shape[0]-1,:]
        avg_data["iteration"].append(avg_df.iteration.mean())
        avg_data["loss_ema"].append(avg_df.loss_ema.mean())
        avg_data["loss"].append(avg_df.loss.mean())
    return pd.DataFrame(avg_data)

def plot_avg_epoch_loss(df, save_path=None):
    '''
    Scatter plot average loss_ema and loss for each epoch
    '''
    font = {
        'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 14
    }

    plt.rc('font', **font)

    fig, axs=plt.subplots(1,2, figsize=(10,5))

    plt.subplots_adjust(wspace=0.3)

    fig.suptitle("Average Training Loss for each Epoch")
    fig.patch.set_facecolor("white")

    axs[0].scatter(
        x=df.index,
        y=df.loss_ema
    )

    axs[1].scatter(
        x=df.index,
        y=df.loss
    )

    axs[0].set_ylabel("EMA Loss")
    axs[0].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")

    if save_path:
        fig.savefig(save_path)
    
    fig.show()

def plot_loss(df, save_path=None):
    '''
    Scatter plot df.loss_ema and df.loss across minibatch iterations
    '''
    font = {
        'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 14
    }

    plt.rc('font', **font)

    fig, axs=plt.subplots(1,2, figsize=(10,5))

    plt.subplots_adjust(wspace=0.3)

    fig.suptitle("Average Loss over minibatch iterations")
    fig.patch.set_facecolor("white")

    axs[0].scatter(
        x=df.iteration,
        y=df.loss_ema
    )

    axs[1].scatter(
        x=df.iteration,
        y=df.loss
    )

    axs[0].set_ylabel("EMA Loss")
    axs[0].set_xlabel("Minibatch iterations")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Minibatch iterations")

    if save_path:
        fig.savefig(save_path)
    
    fig.show()

def collate_train(batch):
    """ collate lists of samples into batches, create [ batch_sz x agent_sz x seq_len x feature] """
    inp = [np.dstack([scene['p_in'], scene['v_in']]) for scene in batch]
    out = [np.dstack([scene['p_out'], scene['v_out']]) for scene in batch]
    inp = torch.FloatTensor(inp)
    out = torch.FloatTensor(out)
    return [inp, out]

def collate_val(batch):
    '''
    Collate function for validation set. Same as `my_collate` except it has no out data
    create [ batch_sz x agent_sz x seq_len x feature]
    '''
    inp = [np.dstack([scene['p_in'], scene['v_in']]) for scene in batch]
    scene_idx=[np.dstack([scene['scene_idx']]) for scene in batch]
    track_ids=[np.dstack([scene['track_id']]) for scene in batch]
    agent_ids=[np.dstack([scene['agent_id']]) for scene in batch]
    inp = torch.FloatTensor(inp)
    return [track_ids, agent_ids, scene_idx, inp]

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
    #                 td=time_data.detach().numpy()
                    data_in['x_pos'].append(time_data[0])
                    data_in['y_pos'].append(time_data[1])
                    data_in['x_vel'].append(time_data[2])
                    data_in['y_vel'].append(time_data[3])
        
        for scene in out_data:
            for car in scene:
                for time_data in car:
    #                 td=time_data.detach().numpy()
                    data_out['x_pos'].append(time_data[0])
                    data_out['y_pos'].append(time_data[1])
                    data_out['x_vel'].append(time_data[2])
                    data_out['y_vel'].append(time_data[3])
        
    #     print(len(train_data['x_pos_in']))
    #     print(len(train_data['x_pos_out']))
        
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

def run_trained_model(model, dataset, batch_size, device):
    '''
    Run trained PyTorch model
    ----------
    Parameters
    ----------
        model - PyTorch Neural Net Model
            Trained PyTorch model to make prediction
        dataset - PyTorch dataset object
            Dataset object to run model on
        batch_size - int
            batch size for DataLoader
        device - str
            device model is on. Options are either 'cuda' or 'cpu'
    -------
    Returns
    -------
        A 2d list of containing the predictions for 60 cars for each scene. 
        list of shape (3200,61)
    '''
    # dataset=ArgoverseDataset(data_path=data_path)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size, 
        shuffle = False, 
        collate_fn=collate_val, 
        num_workers=multiprocessing.cpu_count()
    )

    out_data=[]
    scenes=[]

    model.eval()
    with torch.no_grad():
       for track_ids, agent_ids, scene_idxs, inp in tqdm(data_loader):
            inp=inp.to(device)

            indices=[get_agent_idx(agent_ids[i], track_ids[i]) for i in range(len(agent_ids))]
            indices=torch.tensor(indices, dtype=torch.int).to(device)

            pred=model(inp.reshape(len(inp), -1)).reshape((-1, 60, 30, 4))
            pos_idx=torch.tensor([0,1]).to(device)
            positions=torch.index_select(pred.to(device), 3, pos_idx)
            agent_pos=torch.zeros([batch_size,1,30,2])
            for i in range(indices.shape[0]):
                agent_pos[i]=positions[i][indices[i]]
            agent_pos=agent_pos.reshape(batch_size,60)
            out_data.append(agent_pos.detach().cpu().numpy())

            scene_idxs=np.array(scene_idxs).reshape(1,-1)[0]
            scenes.extend(list(scene_idxs))

    return np.array(scenes), np.array(out_data).reshape(3200,60)

def get_trained_data(model, dataset, batch_size, device):
    '''
    Get DataFrame of predicted outcomes
    ----------
    Parameters
    ----------
        model - PyTorch Neural Net Model
            Trained PyTorch model to make prediction
        dataset - PyTorch dataset object
            Dataset object to run model on
        batch_size - int
            batch size for DataLoader
        device - str
            device model is on. Options are either 'cuda' or 'cpu'
    -------
    Returns
    -------
        Pandas Dataframe of predicted final position for each vehicle in all scenes in <dataset>
    '''
    scenes, out_data=run_trained_model(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        device=device
    )

    df=pd.DataFrame(out_data)
    col_names=["v%i"%i for i in range(1, 61)]
    df.columns=col_names
    df['ID']=scenes
    df=df.loc[:,['ID']+col_names].copy()
    
    return df