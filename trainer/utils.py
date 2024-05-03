import os
import numpy as np
import datetime

import torch
import torch.nn

import torch.distributed as dist

import matplotlib.pyplot as plt


def save(ckpt_dir, net, optim, epoch, model_name="PatchPainting"):
    r"""
    Model Saver

    Inputs:
        ckpt_dir   : (string) check point directory
        netG       : (nn.module) Generator Network
        opitmG     : (torch.optim) Generator's Optimizers
        epoch      : (int) Now Epoch
        model_name : (string) Saving model file's name
    """
    if hasattr(net, "module"):
        netG_dicts = net.module.state_dict()
        try:
            optimG_dicts = optim.module.state_dict()
        except:
            optimG_dicts = optim.state_dict()
    else:
        netG_dicts = net.state_dict()
        optimG_dicts = optim.state_dict()

    torch.save({"net": netG_dicts,
                "optim" : optimG_dicts},
                os.path.join(ckpt_dir, model_name, f"{model_name}_{epoch}.pth"))

def load_gen(ckpt_dir,  netG,  optimG, name, epoch=None, gpu=None):
    r"""
    Model Lodaer

    Inputs:
        ckpt_dir : (string) check point directory
        netG     : (nn.module) Generator Network
        opitmG   : (torch.optim) Generator's Optimizers
        step     : (int) find step.  if NOne, last scale

    """
    ckpt_lst = os.listdir(ckpt_dir)

    if epoch is not None:
        ckptFile = os.path.join(ckpt_dir, name+f"_{epoch}.pth")
    else:
        ckpt_lst.sort()
        ckptFile = os.path.join(ckpt_dir, ckpt_lst[-1])

    if not os.path.exists(ckptFile):
        raise ValueError(f"Please Check Check Point File Path or Epoch, File is not exists!")

    # Load Epochs Now
    epoch = int(ckpt_lst[-1].split("_")[-1][:-4])

    # Load Model 
    if gpu is not None:
        dist.barrier()
        mapLocation = {"cuda:0": f"cuda:{gpu}"}
        dict_model = torch.load(ckptFile, map_location=mapLocation)
    else:
        dict_model = torch.load(ckptFile)
    
    try:
        netG.load_state_dict(dict_model['netG'])
    except:
        netG.module.load_state_dict(dict_model['netG'])

    optimG.load_state_dict(dict_model["optimG"])

    return netG,  optimG, epoch



def plot_recon_figures(sample, pred, output_path, step, num_figures = 5, save=False):

    fig, axs = plt.subplots(num_figures, 3, figsize=(30,15))
    fig.tight_layout()
    axs[0,0].set_title('Ground-truth')
    axs[0,1].set_title('Reconstruction')

    for ax in axs:

        cor = np.corrcoef([pred, sample])[0,1]

        x_axis = np.arange(0, sample.shape[-1])
        # groundtruth
        ax[0].plot(x_axis, sample)
       
        # pred
        ax[1].plot(x_axis, pred)
        ax[1].set_ylabel('cor: %.4f'%cor, weight = 'bold')
        ax[1].yaxis.set_label_position("right")

    fig_name = f'reconst-{datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}-{step}.png'
    if save:
        fig.savefig(os.path.join(output_path, fig_name))
    plt.close(fig)
    return fig