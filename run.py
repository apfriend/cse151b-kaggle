import json
import os
import torch
from torch.utils.data import DataLoader
import argparse
import multiprocessing
from tqdm.auto import tqdm
import numpy as np
#####LOCAL SCRIPTS#####
# os.chdir('./src')
# print(os.getcwd())
import src.main as main
# import models.lstm as lstm
# import models.linear as linear
# import models.physics as physics
# import utility as utility
# from ArgoverseDataset import ArgoverseDataset
# import load_data as load_data
# import run_model as run_model

CONFIG_PATH="config/config.json"

if __name__=="__main__":
    
    with open(CONFIG_PATH, "r") as config_file:
        config_args=json.load(config_file)

    models=config_args["models"]
    train_path=config_args["train_path"]
    val_path=config_args["val_path"]
    device=config_args["device"]
    train_save_path=config_args["train_save_path"]
    model_path=config_args["model_path"]
    submit_path=config_args["submit-path"]

    parser=argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="PyTorch Nerual Net Model to choose", action='store_true')
    parser.add_argument("--epochs", help="Number of epochs to run model", type=int, action='store_true')
    parser.add_argument("-v", "--verbose", help="Print every ten iterations. No Progress bar.", action="store_true")
    parser.add_argument('--submit', help='run trained models to create submission .csv folder', action='store_true')
    parser.add_argument('--train', help='train model based on parameters passed in command line', action='store_true')
    args=parser.parse_args()

    if args.submit:
        models=[os.path.join(model_path,model) for model in models]
        main.run_all(models, val_path, submit_path)
    else:

        if args.model=="lstm":
            my_model=lstm.LSTM_model
            train_model=lstm.train
        elif args.model=="simple-linear":
            my_model=linear.simple_model
            train_model=linear.train
        elif args.model=="multilayer-liner":
            my_model=linear.multilayer_model
            train_model=linear.train
        elif args.model=="multilayer-liner2":
            my_model=linear.multilayer_model2
            train_model=linear.train
        else:
            raise ValueError("Not a valid model!")

        train_dataset  = ArgoverseDataset(data_path=train_path)    

        train_loader=DataLoader(
            dataset=train_dataset,
            batch_size=batch_sz,
            shuffle=True,
            collate_fn=utility.collate_train,
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