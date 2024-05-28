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



def plot_recon_figures(sample, pred, output_path, num_figures = 5, name="none", save=False):

    fig, axs = plt.subplots(num_figures, 3, figsize=(30,15))
    fig.tight_layout()
    axs[0,0].set_title('Ground-truth')
    axs[0,1].set_title('Reconstruction')
    axs[0,2].set_title('Comparison')

    for ax, s, p in zip(axs, sample, pred):

        # cor = np.corrcoef([pred, sample])[0,1]

        x_axis = np.arange(0, sample.shape[1])
        # groundtruth
        ax[0].plot(x_axis, s)
       
        # pred
        ax[1].plot(x_axis, p)
        # ax[1].set_ylabel('cor: %.4f'%cor, weight = 'bold')
        # ax[1].yaxis.set_label_position("right")

        ax[2].plot(x_axis, s, "b")
        ax[2].plot(x_axis, p, "r")
    
    if save:
        fig_name = f'{name}.png'
        fig.savefig(os.path.join(output_path, fig_name))
    plt.close(fig)
    return fig


def plot_recon_figure(sample, pred, output_path, num_figures = 5, name="none", save=False):

    fig, axs = plt.subplots(num_figures, 3, figsize=(20,15))
    fig.tight_layout()
    axs[0].set_title('Ground-truth')
    axs[1].set_title('Reconstruction')
    axs[2].set_title('Comparison')

    

    x_axis = np.arange(0, sample.shape[1])
    # groundtruth
    axs[0].plot(x_axis, sample[0])
    
    # pred
    axs[1].plot(x_axis, pred[0], "--")
    # ax[1].set_ylabel('cor: %.4f'%cor, weight = 'bold')
    # ax[1].yaxis.set_label_position("right")

    axs[2].plot(x_axis, sample[0], "b")
    axs[2].plot(x_axis, pred[0], "r--")

    if save:
        fig_name = f'{name}.png'
        fig.savefig(os.path.join(output_path, fig_name))
    plt.close(fig)
    return fig


def plot_recon_figures_bychannels(sample, pred,  output_path, name, save=True):

    for idx, (s, p) in enumerate(zip(sample[0].T, pred[0].T)):
        fig, ax = plt.subplots(1, 3, figsize=(15,12))
        fig.tight_layout()
        ax[0].set_title('Ground-truth')
        ax[1].set_title('Reconstruction')
        ax[2].set_title('Comparison')

        x_axis = np.arange(0, sample.shape[1])
        ax[0].plot(x_axis, s, "b")    
        ax[1].plot(x_axis, p, "r:")
        ax[2].plot(x_axis, s, "b")
        ax[2].plot(x_axis, p, "r:")


        if save:
            fig_name =  f'{name}_{idx}_channels.png'
            fig.savefig(os.path.join(output_path, fig_name))
        plt.close(fig)
