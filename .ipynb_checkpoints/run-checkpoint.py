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
        collate_fn=load_data.collate_val, 
        num_workers=multiprocessing.cpu_count()
    )

    out_data=[]
    scenes=[]

    model.eval()
    with torch.no_grad():
       for track_ids, agent_ids, scene_idxs, inp in tqdm(data_loader):
            inp=inp.to(device)

            indices=[utils.get_agent_idx(agent_ids[i], track_ids[i]) for i in range(len(agent_ids))]
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