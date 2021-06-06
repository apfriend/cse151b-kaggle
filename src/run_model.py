import os
# import multiprocessing
import numpy as np
import pandas as pd
import torch
# from torch.utils.data import DataLoader
from tqdm import tqdm
#####LOCAL SCRIPTS#####
import utility
import load_data
import models.lstm as lstm
import models.linear as linear
import models.physics as physics

def model_predict(model, data, physics_model=False):
    '''
    Predict using model
    ----------
    Parameters
    ----------
        model - torch nerual network linear model
            model to predict
        data - torch.FloatTensor object
            data to predict using <model>
        physics_model - bool. Default False
            If true do not reshape data for physics model
    -------
    Returns
    -------
        torch.FloatTensor
        predicted outputs of <model>(<data>)        
    '''
    if not physics_model:
        data=data.reshape((len(data),-1))#.reshape((-1, 60, 30, 4))
        out=model(data)
        out=out.reshape((-1,60,30,4))
    else:
        out=model(data)
    return out
    # if isinstance(model, lstm.LSTM_model):
    #     data=data[:,:-1]
    #     out=model(data)[:,-30:]
    #     out=out.reshape((-1,30,60,4)).transpose(1,2)
    # elif isinstance(model, physics.avg_velocity_model):
    #     data=data.reshape((len(data),-1))#.reshape((-1, 60, 30, 4))
    #     out=model(data)
    #     out=out.reshape((-1,60,30,4))
    # else:
    #     out=model(data)
    # return out

def train_evaluate(model, train_path, test_path, loss_data_fp, fig_fp, epochs, batch_size, train_func, 
        ema_weight, plot_size=50, dropout=None, epoch_loss=False, normalize=False, device='cuda:0',
        save_model_path=None, verbose=False):
    '''
    Train PyTorch model and evaluate loss of model.
    ----------
    Parameters
    ----------
        model - torch nerual network linear model
            model to train
        train_path - str
            path to training data
        test_path - str
            path to test data
        loss_data_fp - string
            filepath to save training data to
        fig_fp - string
            filepath to save loss plot to
        epochs - int
            number of epochs to train
        batch_size - int
            training minibatch size
        train_func - python function
            training function to train model
        ema_weight - float
            float between (0.0,1.0) for the exponential moving average weight
        plot_size - int. Default 50
            Number of datapoints to take average of for loss plot
        dropout - float. Default none
            float between (0.0,1.0) for dropout. If None use all training points in training.
        epoch_loss - bool. Default False
            If true plot average loss by epoch instead of averaged by iteration.
        normalize - bool. Default False
            If true normalize data when training.
        device - string, default 'cuda:0'
            choose to run on gpu ('cuda') or cpu ('CPU')
        save_model_path - str, default None
            Path to model to. If set to default None, model will not be saved.
        verbose - boolean, default False
            If true print training progress every 10 training iterations
    -------
    Returns
    -------
        Trained PyTorch model
    '''
    data_loader=load_data.get_data_loaders(train_path, batch_size)
    test_data=load_data.get_test_data(test_path, batch_size)

    loss_plot_fp=fig_fp[:-4]+'-loss.pdf'

    trained_model, loss_df=train_func(
        model=model,
        data_loader=data_loader,
        test_data=test_data,
        n_epochs=epochs,
        filename=loss_data_fp,
        ema_weight=ema_weight,
        dropout=dropout,
        batch_sz=batch_size,
        device=device,
        verbose=verbose,
        normalize=normalize
    )

    if save_model_path:
        torch.save(trained_model.state_dict(), save_model_path)
    
    if epoch_loss:
        avg_loss=utility.get_avg_epoch_loss(loss_df)
        utility.plot_avg_epoch_loss(avg_loss, loss_plot_fp)
    else:
        avg_loss=utility.get_avg_loss(loss_df, plot_size)
        utility.plot_loss(avg_loss, loss_plot_fp)

    trajectory_plot_fp=fig_fp[:-4]+'-trajectory.pdf'
    test_inp, test_out=test_data
    test_inp, test_out=test_inp.to(device), test_out.to(device)
    test_preds=model_predict(trained_model, test_inp)
    utility.visualize_trajectory(test_inp, test_out, test_preds)

    return trained_model

def run_model_submit(model, data_path, batch_size, physics_model=False, device='cuda:0'):
    '''
    Run trained PyTorch model on Kaggle validation data for submition.
    ----------
    Parameters
    ----------
        model - PyTorch Neural Net Model
            Trained PyTorch model to make prediction
        data_path - str
            Path to validation data
        batch_size - int
            batch size for DataLoader
        physics_model - bool. Default False
            If true run on basic non neural net physics model
        device - str. Default: cuda:0
            device model is on. Options are either 'cuda' or 'cpu'
    -------
    Returns
    -------
        A 2d list of containing the predictions for 60 cars for each scene. 
        list of shape (3200,61)
    '''
    data_loader=load_data.get_data_loaders(data_path, batch_size, val_data=True)

    out_data=[]
    scenes=[]
    if not physics_model:
        model.eval()
    with torch.no_grad():
       for track_ids, agent_ids, scene_idxs, inp in tqdm(data_loader):
            inp=inp.to(device)

            indices=[utility.get_agent_idx(agent_ids[i], track_ids[i]) for i in range(len(agent_ids))]
            indices=torch.tensor(indices, dtype=torch.int).to(device)
            
            if physics_model:
                pred=model_predict(model, inp, physics_model=physics_model)
                positions=pred
            else:
                pred=model_predict(model, inp)
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

def submit_model(model, data_path, batch_size, save_path, save_fn, physics_model=False, device='cuda:0'):
    '''
    Create submission dataset and save to csv
    ----------
    Parameters
    ----------
        model - PyTorch Neural Net Model
            Trained PyTorch model to make prediction
        data_path - str
            Path to validation data
        batch_size - int
            batch size for DataLoader
        save_path - str
            path to directory to save submission data to
        save_fn - str 
            filename to save submission data as (must be .csv file)
        physics_model - bool. Default False
            If true run on basic non neural net physics model
        device - str. Default: cuda:0z
            device model is on. Options are either 'cuda' or 'cpu'
    -------
    Returns
    -------
        Pandas DataFrame of submission data
    '''
    scenes, submit_preds=run_model_submit(
        model=model,
        data_path=data_path,
        batch_size=batch_size,
        physics_model=physics_model,
        device=device
    )

    col_names=['ID']+['v%i'%i for i in range(1,61)]
    submit_data=np.concatenate([scenes.reshape(-1,1), submit_preds], axis=1)

    submit_df=pd.DataFrame(
        data=submit_data,
        columns=col_names
    )

    submit_df.ID=submit_df.ID.astype(np.int32)

    fn=os.path.join(save_path, save_fn)
    submit_df.to_csv(fn, index=False)

    return submit_df

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
    scenes, out_data=run_model_submit(
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