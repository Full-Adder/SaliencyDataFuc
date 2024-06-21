import os 
import sys
import platform

import random
import numpy as np

import torch
import torchvision
import torchaudio

from dataset import get_dataloader

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

saliency_path = r"/home/wjq/workplace/data/saliency_data/"


if __name__ == "__main__":
    task_list = ['class_increase', 'domain_increase']
    print(f"use {platform.system()}, torch {torch.__version__}, torchvision {torchvision.__version__}, torchaudio {torchaudio.__version__}")
    for task in task_list:
        data_loader = get_dataloader(root=saliency_path, mode='train', task=task)
        print(f"{task= }, {len(data_loader)= }")
        for i,(data, target, vail) in enumerate(data_loader):
            print(i, data['rgb'].shape, data['audio'].shape, target['salmap'].shape, target['binmap'].shape)
            break


