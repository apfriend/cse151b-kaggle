import os
import sys
import json
import pickle
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader
import argparse
import models.lstm as lstm
import models.linear as linear

CONFIG_PATH="config/config.json"

def my_collate(batch):
    """ collate lists of samples into batches, create [ batch_sz x agent_sz x seq_len x feature] """
    inp = [np.dstack([scene['p_in'], scene['v_in']]) for scene in batch]
    out = [np.dstack([scene['p_out'], scene['v_out']]) for scene in batch]
    inp = torch.FloatTensor(inp)
    out = torch.FloatTensor(out)
    return [inp, out]

def my_collate2(batch):
    '''
    Collate function for validation set. Same as `my_collate` except it has no out data
    create [ batch_sz x agent_sz x seq_len x feature]
    '''
    inp = [np.dstack([scene['p_in'], scene['v_in']]) for scene in batch]
    inp = torch.FloatTensor(inp)
    return inp

class ArgoverseDataset(Dataset):
    """Dataset class for Argoverse"""
    def __init__(self, data_path: str, transform=None):
        super(ArgoverseDataset, self).__init__()
        self.data_path = data_path
        self.transform = transform

        self.pkl_list = glob(os.path.join(self.data_path, '*'))
        self.pkl_list.sort()
        
    def __len__(self):
        return len(self.pkl_list)

    def __getitem__(self, idx):

        pkl_path = self.pkl_list[idx]
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            
        if self.transform:
            data = self.transform(data)

        return data

if __name__=="__main__":
    
    with open(CONFIG_PATH, "r") as config_file:
        config_args=json.load(config_file)

    batch_sz=config_args["batch_size"]
    n_workers=config_args["num_workers"]
    train_path=config_args["train_path"]
    val_path=config_args["val_path"]
    device=config_args["device"]
    train_save_path=config_args["train_save_path"]
    ema_weight=config_args["ema_weight"]

    parser=argparse.ArgumentParser()
    parser.add_argument("--model", help="PyTorch Nerual Net Model to choose")
    parser.add_argument("--epochs", help="Number of epochs to run model", type=int)
    parser.add_argument("-v", "--verbose", help="Print every ten iterations. No Progress bar.", action="store_true")
    args=parser.parse_args()

    if args.model=="lstm":
        my_model=lstm.LSTM_model
        train_model=lstm.train
    elif args.model=="simple-linear":
        my_model=linear.simple_model
        train_model=linear.train
    elif args.model=="multilayer-liner":
        my_model=linear.multilayer_model
        train_model=linear.train
    else:
        raise ValueError("Not a valid model!")

    train_dataset  = ArgoverseDataset(data_path=train_path)
    val_dataset=ArgoverseDataset(data_path=val_path)

    train_loader=DataLoader(
        train_dataset,
        batch_size=batch_sz,
        shuffle=True,
        collate_fn=my_collate,
        num_workers=n_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_sz, 
        shuffle = False, 
        collate_fn=my_collate2, 
        num_workers=n_workers
    )

    train_model(
        model=my_model,
        data_loader=train_loader,
        n_epochs=args.epochs,
        filename=train_save_path,
        ema_weight=ema_weight,
        device=device,
        verbose=args.verbose
    )