import json
import torch
from torch.utils.data import DataLoader
import argparse
import src.models.lstm as lstm
import src.models.linear as linear
import src.util_functions as utils
from src.ArgoverseDataset import ArgoverseDataset
import multiprocessing
from tqdm import tqdm

CONFIG_PATH="config/config.json"


if __name__=="__main__":
    
    with open(CONFIG_PATH, "r") as config_file:
        config_args=json.load(config_file)

    batch_sz=config_args["batch_size"]
    train_path=config_args["train_path"]
    val_path=config_args["val_path"]
    device=config_args["device"]
    train_save_path=config_args["train_save_path"]
    ema_weight=config_args["ema_weight"]

    parser=argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="PyTorch Nerual Net Model to choose")
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

    train_loader=DataLoader(
        dataset=train_dataset,
        batch_size=batch_sz,
        shuffle=True,
        collate_fn=utils.collate_train,
        num_workers=multiprocessing.cpu_count()
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