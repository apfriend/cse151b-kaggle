import json
import os
import torch
from torch.utils.data import DataLoader
import argparse
import multiprocessing
from tqdm.auto import tqdm
import numpy as np
#####LOCAL SCRIPTS#####
import src.models.lstm as lstm
import src.models.linear as linear
import src.models.physics as physics
import src.utility as utility
from src.ArgoverseDataset import ArgoverseDataset
import src.load_data as load_data
import src.run_model as run_model

def run_all(models, src, dst):
    '''
    Run all models set in config file
    ----------
    Parameters
    ----------
        models - list
            list of paths to trained PyTorch models
        dst - str
            path to save submit csv to
    '''
    models_objects=[linear.simple_model, linear.multilayer_model, linear.multilayer_model2, physics.avg_veloctiy_model]
    model_names=['simple linear','multilayer linear','multilayer linear2', 'physics-model-submit']
    model_lookup=dict(zip(model_names, models_objects))

    for model in tqdm(models):
        if 'physics' not in model:
            model_name, dropout, batch_size, epochs=utility.get_model_data(model) 
            is_physics=False
        else:
            model_name='physics-model-submit-test'
            batch_size=128
            is_physics=True

        submit_fn=model.replace('.pth','.csv')

        run_model.submit_model(
            model=model_lookup[model_name],
            data_path=src,
            batch_size=batch_size,
            save_path=dst,
            save_fn=submit_fn,
            physics_model=is_physics
        )