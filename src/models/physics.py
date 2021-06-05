import torch
import pandas as pd
import numpy as np
from torch import nn
from tqdm import tqdm
from time import time
from IPython.display import display
#####LOCAL SCRIPTS#####
import utility

def avg_velocity_model(inp, device='cuda:0'):
    '''
    Calculate future positions based on average velocity added to final position
    ----------
    Parameters
    ----------
        inp - torch.FloatTensor object
            Tensor of input positions and velocities for 60 cars at 19 instances, each 0.1 seconds apart.
            dimensions: (batch_size, 60, 19, 4)
            The first 2 items of the 4th dimension are the positions, and the last 2 items are the velocities
        device - string, default 'cuda:0'
            choose to run on gpu ('cuda') or cpu ('CPU')
    '''
    #save shapes if input tensor
    # print('\nPhyiscs model')
    # print(inp.shape)
    batch_size, num_agents=inp.shape[0],inp.shape[1]
    num_final_pos=30 #number of time x,y positions to predict

    #indexes to separate position and velocity from input tensor
    pos_idx=torch.tensor([0,1]).to(device)
    vel_idx=torch.tensor([2,3]).to(device)

    # print(vel_idx)

    #make velocity and position tensors
    vel_tensor=torch.index_select(inp, 3, vel_idx)
    pos_tensor=torch.index_select(inp, 3, pos_idx)

    #compute average velocity across all input time 
    avg_vel_tensor=torch.mean(
        input=vel_tensor,
        dim=2,
        keepdim=True
    )

    #get the final position from position tensor
    final_pos_idx=torch.tensor([inp.shape[2]-1]).to(device)
    final_pos_tensor=torch.index_select(pos_tensor, 2, final_pos_idx)

    #indices of x and y coordinates
    x_idx=torch.tensor([0]).to(device)
    y_idx=torch.tensor([1]).to(device)

    #get final x and y positions and velocities and reshape to be 2d tensor
    final_x_pos_tensor=torch.index_select(final_pos_tensor, 3, x_idx).reshape(batch_size*num_agents, 1)
    final_y_pos_tensor=torch.index_select(final_pos_tensor, 3, y_idx).reshape(batch_size*num_agents, 1)
    avg_x_vel_tensor=torch.index_select(avg_vel_tensor, 3, x_idx).reshape(batch_size*num_agents, 1)
    avg_y_vel_tensor=torch.index_select(avg_vel_tensor, 3, y_idx).reshape(batch_size*num_agents, 1)

    #make tensor of integers [1,30] to multiply agains velocity
    time_tensor=torch.FloatTensor([list(range(1,num_final_pos+1))]).to(device)

    #get change in position by multiplying velocity tensors by time tensor
    x_pos_delta_tensor=torch.tensordot(
        avg_x_vel_tensor, 
        time_tensor, 
        dims=([1],[0])
    )    
    y_pos_delta_tensor=torch.tensordot(
        avg_y_vel_tensor, 
        time_tensor, 
        dims=([1],[0])
    )

    #add change in position tensors to final input position tensor to get final output positions
    final_x_pos_tensor=torch.cat([final_x_pos_tensor]*num_final_pos, dim=1)
    final_y_pos_tensor=torch.cat([final_y_pos_tensor]*num_final_pos, dim=1)

    final_x_pos_tensor=final_x_pos_tensor+x_pos_delta_tensor
    final_y_pos_tensor=final_y_pos_tensor+y_pos_delta_tensor

    #join x and y positions into final output tensor
    final_pos=torch.zeros(batch_size*num_agents, num_final_pos*2)
    final_pos[:,::2]=final_x_pos_tensor
    final_pos[:,1::2]=final_y_pos_tensor
    # print("\nshape before reshape: ",final_pos.shape)
    final_pos=final_pos.reshape(batch_size, num_agents, num_final_pos, 2).to(device)
    # print("\nshape after reshape: ",final_pos.shape)
    return final_pos