import gc, os
import numpy as np

import torch
from torch import nn

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


from models import eeg_AutoEncoder
from trainer.base_trainer import BaseTrainer
from trainer.utils import plot_recon_figures, save
from libs.losses import l1, l2, SignalDiceLoss
from libs.metric import SignalDice

class EEGAETrainer(BaseTrainer):
    def __init__(self, json_file, sharedFilePath):
        self.json_dict = self.json_load(json_file)
        self.sharedFilePath = sharedFilePath
        self.startEpoch     = 0
        self.globalStep     = 0
        super().__init__(self.json_dict["ckpt_dir"], self.json_dict["log_dir"], self.json_dict["batch_size"], 
                         sharedFilePath, self.json_dict["num_workers"])

        self.json_parser()
        self.__checkDirectory__()
        self.losses = {"sdsc":[], "rec":[], "loss":[]}
        self.accuracy = {"sdsc":[], "mse":[]}

        
    def json_parser(self):

        ########################################
        #            Train Setting
        ########################################
        self.name               = self.json_dict["name"]
        self.data_name          = self.json_dict["data_name"]
        self.log_dir            = self.json_dict["log_dir"]
        self.ckpt_dir           = self.json_dict["ckpt_dir"]
        self.eeg_train_path     = self.json_dict["eeg_train_path"]
        self.eeg_test_path      = self.json_dict["eeg_test_path"]
        self.eeg_val_path       = self.json_dict["eeg_val_path"]
        self.img_path           = self.json_dict["img_path"]
        self.split_path         = self.json_dict["split_path"]


        ########################################
        #            Model Setting
        ########################################
        self.in_seq         = self.json_dict["in_seq"]
        self.in_channels    = self.json_dict["in_channels"]
        self.z_channels     = self.json_dict["z_channels"]
        self.out_seq        = self.json_dict["out_seq"]
        self.dims           = self.json_dict["dims"]
        self.shortcut       = bool(self.json_dict["shortcut"])
        self.dropout        = self.json_dict["dropout"]
        self.groups         = self.json_dict["groups"]
        self.layer_mode     = self.json_dict["layer_mode"]
        self.block_mode     = self.json_dict["block_mode"]
        self.down_mode      = self.json_dict["down_mode"]
        self.up_mode        = self.json_dict["up_mode"]
        self.pos_mode       = self.json_dict["pos_mode"]
        self.n_layer        = self.json_dict["n_layer"]
        self.n_head         = self.json_dict["n_head"]
        self.dff_factor     = self.json_dict["dff_factor"]
        self.stride         = self.json_dict["stride"]
        self.sdsc_lambda    = self.json_dict["sdsc_lambda"]


        ########################################
        #         Training Parameters
        ########################################
        self.lr        = self.json_dict["learningRate"]
        self.epochs    = self.json_dict["trainEpochs"]
        self.saveIter  = self.json_dict["saveIter"]
        self.validIter = self.json_dict["validIter"]
        self.logIter   = self.json_dict["logIter"]
        self.restart   = bool(self.json_dict["restart"])
    
    def model_define(self, gpu):
        self.MODEL = eeg_AutoEncoder(self.in_seq,  self.in_channels, self.z_channels, self.out_seq,    
                                    self.dims, self.shortcut, self.dropout, self.groups, self.layer_mode, 
                                    self.block_mode, self.down_mode, self.up_mode, self.pos_mode,   
                                    self.n_layer, self.n_head, self.dff_factor, self.stride).to(gpu)
        
        self.optim = torch.optim.AdamW(filter(lambda x: x.requires_grad, self.MODEL.parameters()), lr= self.lr, betas=[0, 0.99])
        self.sdsc_loss = SignalDiceLoss(sep=True)
        self.mse  = nn.MSELoss()
        self.sdsc = SignalDice()

        
    def _train(self, gpu, size):
        print(f"Now Initialize Rank: {gpu} | Number Of GPU : {size}")
        self.model_define(gpu)        
        self.initialize(gpu, size)

        self.MODEL = nn.SyncBatchNorm.convert_sync_batchnorm(DDP(self.MODEL))
        self.makeDatasets(self.eeg_train_path, self.eeg_test_path, self.eeg_val_path, self.img_path)

        if gpu == 0:
            self.makeTensorBoard()
            # with open("./log/my_model.log", "w") as f:
            #     result, _ = summary(self.auto_encoder, input_size=(self.heatmap_size, self.joints, self.imageSize, self.imageSize), batch_size=self.batchSize)
            #     f.write(result)

        print(f"DDP RANK {gpu} RUN...")


        for epoch in range(self.startEpoch, self.epochs):
            self.train_dataset_sampler.set_epoch(epoch),
            self.valid_dataset_sampler.set_epoch(epoch)
            torch.cuda.empty_cache()
            gc.collect()

            for step, data in enumerate(self.loader_train):
                self.MODEL.train()
                self.optim.zero_grad()

                eeg, image, label = data
                
                latent, rec = self.MODEL(eeg.to(gpu))

                rec_loss    = torch.mean(torch.sum(l2(rec, eeg), dim=1))
                sdsc_loss   = self.sdsc_loss(rec, eeg)
                loss        =  rec_loss + self.sdsc_lambda * sdsc_loss

                mse         = self.mse(rec, eeg)
                sdsc        = self.sdsc(rec, eeg)

                self.losses["sdsc"].append(sdsc_loss.item())
                self.losses["rec"].append(rec_loss.item())
                self.losses["loss"].append(loss.item())

                self.accuracy["mse"].append(mse.item())
                self.accuracy['sdsc'].append(sdsc.item())

                loss.backward()
                self.optim.step()

                if self.globalStep % self.logIter == 0:
                    # LOG Print
                    strings = f"Train Step {self.globalStep} | LOSS {np.mean(self.losses['loss']):.4f} | SDSC LOSS {np.mean(self.losses['sdsc']):.4f} | RECON LOSS {np.mean(self.losses['rec']):.4f}"
                    strings += f" Accuracy MSE {np.mean(self.losses['mse']):.4f} | SDSC {np.mean(self.losses['sdsc']):.4f}"
                    print(strings)
                    
                    # Tensorboard logged
                    self.summaryWriter.add_scalar("LOSS", np.mean(self.losses['loss']), self.globalStep)
                    self.summaryWriter.add_scalar("SDSC LOSS", np.mean(self.losses['sdsc']), self.globalStep)
                    self.summaryWriter.add_scalar("RECON LOSS", np.mean(self.losses['rec']), self.globalStep)

                    self.summaryWriter.add_scalar("MSE", np.mean(self.accuracy['mse']), self.globalStep)
                    self.summaryWriter.add_scalar("SDSC", np.mean(self.accuracy['sdsc']), self.globalStep)

                    # Draw Reconstruction
                    fig = plot_recon_figures(eeg.to('cpu').numpy(), rec.to('cpu').numpy(), self.log_dir, self.globalStep)

                    self.summaryWriter.add_figure("EEG Recon", fig, self.globalStep)

                self.globalStep += 1
    
    def checkPoint(self, gpu):
        if gpu == 0:
            save(self.ckpt_dir, self.MODEL, self.optim, self.globalStep, self.name)
        dist.barrier()
        mapLocation = {"cuda:0": f"cuda:{gpu}"}
        dict_model = torch.load(os.path.join(self.ckpt_dir, self.name, f"{self.name}_{self.globalStep}.pth"), map_location=mapLocation)
        self.MODEL.module.load_state_dict(dict_model["net"])
    
    def runTrain(self):
        gpus = torch.cuda.device_count()
        mp.spawn(self._train, args=(gpus,), nprocs=gpus)