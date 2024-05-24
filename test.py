import argparse
import torch, os
import torch.nn as nn

from trainer.eeg_ae_trainer import EEGAETrainer as trainer


parser = argparse.ArgumentParser(description="Evaluation of Patch Painting Transformer",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c', '--config', default= "./config/EEGAE_Test.json", help="Model's Config", type=str, dest="config")
parser.add_argument("-s", "--shared_path", default=os.path.dirname(os.path.abspath(__file__)), help="Shared File Path For Distributed Learning", dest="shared")

args = parser.parse_args()

configFilePath = args.config
sharedFilePath = args.shared

if __name__ =="__main__":
    Trainer = trainer(configFilePath, sharedFilePath)
    Trainer.test(folder_name="EEG_condition_SMSE")