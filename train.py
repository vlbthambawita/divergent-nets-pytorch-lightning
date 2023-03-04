#=========================================================
# Developer: Vajira Thambawita (PhD)
# References: * https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/binary_segmentation_intro.ipynb
#             * https://pytorch-lightning.readthedocs.io/en/stable/
#=========================================================



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



#==========================================
# Prepare Data
#==========================================



# Pytorch lightning training
class PolypModel(pl.LightningModule):

    def __init__(self, arch="UnetPlusPlus", encoder_name= "resnet34", in_channels=3, out_classes=2, lr=0.0001, test_print_batch_id=0, test_print_num=5, **kwargs):
        super().__init__()
        
        # "UnetPlusPlus", "resnet34", in_channels=3, out_classes=2
        self.arch = arch
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.out_classes = out_classes
        self.test_print_batch_id = test_print_batch_id
        self.test_print_num = test_print_num
        self.lr = lr
        

        self.model = smp.create_model(
            self.arch, encoder_name=self.encoder_name, in_channels=self.in_channels, classes=self.out_classes, **kwargs)

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1)) # self.std
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1)) # self.mean

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        
        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        #print(f"h={h}, w={w}")
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset 
        # with "empty" images (images without target class) a large gap could be observed. 
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        
        if stage == "test":
            print(metrics)
        
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        image = batch["image"]
        image_origin = batch["image_origin"]
        mask = batch["mask"]
        
        image = (image - self.mean) / self.std
        mask_p = self.model(image)
        
        prob_mask = mask_p.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        
        loss = self.loss_fn(mask_p, mask)
        print(mask_p.shape)
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")
        
        return {"image_origin": image_origin, 
                "mask": mask, 
                "prob_mask": prob_mask, 
                "pred_mask": pred_mask,
                "loss": loss,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,}  

    def test_epoch_end(self, outputs):
        # test images from the first batch of test data
        #print(outputs[0]["image_origin"][0].shape)
        #print(outputs[0]["pred_mask"][0].shape)
        #print(outputs[0]["image_origin"][0].max())
        #print(outputs[0]["image_origin"][0].min())
        
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        
        
        for i in range(self.test_print_num):
            img = outputs[0]["image_origin"][i].cpu().numpy()
            mask = outputs[0]["pred_mask"][i][0,:, :].cpu().numpy()
            plt.imsave(f"image_{i}.png",img) 
            plt.imsave(f"mask_pred_{i}.png", mask) 
            
        # logger.log_image(key=f"samples_{i}", images=[img, mask])
        #plt.imsave("test_pred_mask_1.png",outputs[0]["image_origin"][0][1,:, :].cpu().numpy())
        #img = Image.fromarray(outputs[0]["image"][0].permute(1,2,0).cpu().numpy())
        #img = img.save("test_img.png")
            
    
    def predict_step(self, batch, batch_idx):
        image = batch["image"]
        mask = batch["mask"]
        image = (image - self.mean) / self.std
        mask_p = self.model(image)
        prob_mask = mask_p.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        print(mask_p.shape)
        
        return {"image": image, "mask": mask, "prob_mask": prob_mask, "pred_mask": pred_mask}
    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    
#================================================
# Train the model
#================================================
def train_model(train_loader, valid_loader, model, checkpoint_callback):
    

       # create epoch runners 
    # it is a simple loop of iterating over dataloader`s samples
    trainer = pl.Trainer(
        gpus=1, 
        max_epochs=5,callbacks=[checkpoint_callback])


    trainer.fit(
    model, 
    train_dataloaders=train_loader, 
    val_dataloaders=valid_loader,
    )
            
    


    
# update here
    

#==============================================
# Heatmap generator from tensor
#==============================================
def generate_heatmapts(img_tensor):
    print(img_tensor.shape)
    fig_list = []
    for n in range(img_tensor.shape[0]):
        img = img_tensor[n]
        img = img.squeeze(dim=0)
        img_np = img.detach().cpu().numpy()
        #img_np = np.transforms(img_np, (1,2,0))
        
        plt.imshow(img_np, cmap="hot")
        fig = plt.gcf()
        fig_list.append(fig)
        # plt.clf()
        plt.close()

    return fig_list



#===============================================
# Prepare models
#===============================================
def prepare_model(opt):
    # model = UNet(n_channels=4, n_classes=1) # 4 = 3 channels + 1 grid encode

    # create segmentation model with pretrained encoder
 
    model = PolypModel("UnetPlusPlus", "resnet34", in_channels=3, out_classes=2)

    return model

#====================================
# Run training process
#====================================
def run_train(conf):
    model = prepare_model(conf)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(conf.encoder, conf.encoder_weights)
    

    train_loader, val_loader = prepare_data(conf, preprocessing_fn=None)
    
    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="valid_dataset_iou",
        mode="max",
        dirpath="output/checkpoints/new/",
        filename="sample-polyp-{epoch:02d}-{valid_dataset_iou:.2f}",
    )

    #loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    #metrics = [
    #    smp.metrics.IoU(threshold=0.5, ignore_channels=[0]),
    #]

   # optimizer = torch.optim.Adam([ 
    #    dict(params=model.parameters(), lr=opt.lr),
    #])

    train_model(train_loader, val_loader, model, checkpoint_callback)
#====================================
# Re-train process
#====================================
def run_retrain(opt):

    checkpoint_dict = torch.load(os.path.join(CHECKPOINT_DIR, opt.best_checkpoint_name))

    opt.start_epoch =  checkpoint_dict["epoch"]
    model = checkpoint_dict["model"]

    print("Model epoch:", checkpoint_dict["epoch"])
    print("Model retrain started from epoch:", opt.start_epoch)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(opt.encoder, opt.encoder_weights)

    train_loader, val_loader = prepare_data(opt, preprocessing_fn)

    loss = smp.utils.losses.DiceLoss()

    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=opt.lr),
    ])

    train_model(train_loader, val_loader, model, loss, metrics, optimizer, opt)

#=====================================
# Check model
#====================================
def check_model_graph():
    raise NotImplementedError


#===================================
# Inference from pre-trained models
#===================================

def do_test(opt):


    checkpoint_dict = torch.load(os.path.join(CHECKPOINT_DIR, opt.best_checkpoint_name))

    test_epoch =  checkpoint_dict["epoch"]
    best_model = checkpoint_dict["model"]

    print("Model best epoch:", test_epoch)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(opt.encoder, opt.encoder_weights)
    test_dataset = prepare_test_data(opt, preprocessing_fn=None)
    test_dataset_vis = prepare_test_data(opt, preprocessing_fn=None)
    
    
    for i in range(opt.num_test_samples):
        image, mask = test_dataset[i]
        image_vis, _ = test_dataset_vis[i]

        #print(image)

        mask_tensor = torch.from_numpy(mask).to(opt.device).unsqueeze(0)

        image_tensor = torch.from_numpy(image).to(opt.device).unsqueeze(0)
        pr_mask = best_model.predict(image_tensor)

        pr_mask = pr_mask.squeeze().cpu().numpy().round()

        fig = visualize(
            input_image_new=np.transpose(image_vis, (1,2,0)).astype(int),
            GT_mask_0=mask[0, :,:],
            Pred_mask_0 = pr_mask[0,:,:],
            GT_mask_1= mask[1,:,:],
            Pred_mask_1 = pr_mask[1, :,:]
        )

        fig.savefig(f"./test_202_{i}.png")
        #writer.add_figure(f"Test_sample/sample-{i}", fig, global_step=test_epoch)
        writer.log({"test_epoch": fig})





def check_test_score(opt):

    

    checkpoint_dict = torch.load(os.path.join(CHECKPOINT_DIR, opt.best_checkpoint_name))

    test_best_epoch =  checkpoint_dict["epoch"]
    best_model = checkpoint_dict["model"]

    print("Model best epoch:", test_best_epoch)
    
    

    preprocessing_fn = smp.encoders.get_preprocessing_fn(opt.encoder, opt.encoder_weights)
    test_dataset = prepare_test_data(opt, preprocessing_fn=preprocessing_fn)
    
    test_dataloader = DataLoader(test_dataset, num_workers=48)

    loss = smp.utils.losses.DiceLoss()
    # Testing with two class layers
    metrics = [
        #smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.IoU(threshold=0.5, ignore_channels=None),
    ]

    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )

    logs = test_epoch.run(test_dataloader)
    print("logs=", str(logs))
    writer.add_text(f"{opt.py_file}-test-score", str(logs), global_step=test_best_epoch)

    # Testing with only class layer 1 (polyps)
    loss = smp.utils.losses.DiceLoss(ignore_channels=[0])
    metrics = [
        #smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0]),
    ]

    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )

    logs = test_epoch.run(test_dataloader)
    print("logs=", str(logs))
    writer.add_text(f"{opt.py_file}-test-score-ignore-channel-0", str(logs), global_step=test_best_epoch)



    # Testing with only class layer 0 (BG)

    loss = smp.utils.losses.DiceLoss(ignore_channels=[1])
    metrics = [
        #smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[1]),
    ]

    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )

    logs = test_epoch.run(test_dataloader)
    print("logs=", str(logs))
    writer.add_text(f"{opt.py_file}-test-score-ignore-channel-1", str(logs), global_step=test_best_epoch)

# Implementing a CLI
def cli_main():
    wandb.finish()
    # Initialize Wandb
    logger = WandbLogger(entity="simulamet_mlc", project="diffusion_polyp")
    # model = PolypModel("UnetPlusPlus", "resnet34", in_channels=3, out_classes=2)
    cli = LightningCLI(PolypModel, PolypDataModule, 
                       save_config_kwargs={"config_filename": "test_config.yaml", 'overwrite':True}
                       )
    


if __name__ == "__main__":

    cli_main()
    wandb.finish()

   

