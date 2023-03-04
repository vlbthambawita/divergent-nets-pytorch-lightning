
import argparse
from omegaconf import OmegaConf
from datetime import datetime
import os
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#Pytorch
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms,datasets, utils
from torchvision.utils import save_image
#from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
#from torchsummary import summary

import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningCLI

#from pytorch_lightning.demos.boring_classes import DemoModel, BoringDataModule


from data.dataset import Dataset
from data.prepare_data import prepare_test_data, PolypDataModule
from data.prepare_data import prepare_data_from_multiple_dirs as prepare_data
#from data import PolypsDatasetWithGridEncoding
#from data import PolypsDatasetWithGridEncoding_TestData
#import pyra_pytorch as pyra
#from utils import dice_coeff, iou_pytorch, visualize

import segmentation_models_pytorch as smp
import wandb
from pytorch_lightning.loggers import WandbLogger


#==========================================
# Tensorboard
#==========================================

logger = WandbLogger(entity="simulamet_mlc", project="diffusion_project")