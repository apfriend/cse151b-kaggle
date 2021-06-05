import os
import torch
import pickle
import random
import numpy as np
import multiprocessing
from glob import glob
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
#####LOCAL SCRIPTS#####
from ArgoverseDataset import ArgoverseDataset

def collate_train(batch):
    '''
    Collate lists of samples into batches, create [ batch_sz x agent_sz x seq_len x feature]
    '''
    inp = [np.dstack([scene['p_in'], scene['v_in']]) for scene in batch]
    out = [np.dstack([scene['p_out'], scene['v_out']]) for scene in batch]
    inp = torch.FloatTensor(inp)
    out = torch.FloatTensor(out)
    return [inp, out]

def collate_train2(batch):
    '''
    Collate lists of samples into batches, create [ batch_sz x agent_sz x seq_len x feature]
    '''
    inp = [np.dstack([scene['p_in'], scene['v_in']]) for scene in batch]
    lane_inp=[np.dstack([scene['p_in'], scene['v_in']]) for scene in batch]
    out = [np.dstack([scene['p_out'], scene['v_out']]) for scene in batch]
    inp = torch.FloatTensor(inp)
    out = torch.FloatTensor(out)
    return [inp, lane_inp, out]

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

def get_test_data(path, test_size=None):
    '''
    Get test dataset
    ----------
    Parameters
    ----------
        path - str
            Path of test data
        test_size - int. Default None
            Size of test data to return. Should equal batch size in training.
            If not passed defaults to none and all data from <path> will be returned.
        include_out - bool. Default False
            If true return test data with predicted values
    -------
    Returns
    -------
        List of Pytorch.FloatTensor objects of test input and output data. See `collate_train` for more detail.
    '''
    files=glob(os.path.join(path, '*'))
    if test_size!=None:
        files=random.sample(files, test_size)
    test_data=[0]*len(files)


    idx=0
    for file in files:
        with open(file, 'rb') as file:
            test_data[idx]=pickle.load(file)
            idx+=1
    
    return collate_train(test_data)
    # return test_data

def get_data_loaders(data_path, batch_size, val_data=False):
    '''
    Get train, test, and validation dataloader objects
    ----------
    Parameters
    ----------
        train_path - str
            path to training data
        batch_size - int
            batch size for Data Loader   
        val_data - bool. Default False
            If true return validation data
    -------
    Returns
    -------
        If <val_path> is not `None` return torch.utils.data.DataLoader objects for both train and validation set.
            (train_loader, val_loader) 
        Otherwise return train data loader
            train_loader
    '''
    dataset=ArgoverseDataset(data_path=data_path)

    if val_data:
        data_loader=DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_val,
            num_workers=multiprocessing.cpu_count()
        )
    else:
        data_loader=DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_train,
            num_workers=multiprocessing.cpu_count()
            # num_workers=1
        ) 
    
    return data_loader

def load_model(src, model, device='cuda:0'):
    '''
    Load trained PyTorch model to disk
    ----------
    Parameters
    ----------
        src - str  
            Valid filepath of saved PyTorch model.
        device - str. Default: `cuda:0`
            Name of device to load model to.
    -------
    Returns
    -------
        torch.nn.Module <model> class with trained state dictionary from disk loaded into <device> 
    '''
    state_dict=torch.load(src)
    model.load_state_dict(state_dict=state_dict)
    model.to(device)
    return model
