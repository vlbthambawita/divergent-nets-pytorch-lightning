from torch.utils.data import Dataset as BaseDataset
import numpy as np
import glob
import os
from PIL import Image


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    
    def __init__(
            self, 
            img_dir,
            mask_dir,
            img_ext = "jpg",
            mask_ext = "jpg",
            classes=[0,255], 
            augmentation=None, 
            preprocessing=None,
            mask_clean=True
    ):
       
        self.img_dir = img_dir
        self.mask_dir  = mask_dir
        
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        
        # Get all image and mask paths using given extensions
        self.images = glob.glob(img_dir + f"/*.{img_ext}") 
        self.masks = glob.glob(mask_dir + f"/*.{mask_ext}") 
        
        # convert str names to class values on masks
        self.class_values = classes
        
        self.mask_clean = mask_clean
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        
        # Getting images and masks without any mismatch
        img_path = self.images[i]
        img_name = img_path.split("/")[-1].split(".")[0]
        mask_path = os.path.join(self.mask_dir, f"{img_name}.{self.mask_ext}")
        img = Image.open(img_path)
        image = np.array(img)
        image_origin = np.array(img.resize((256, 256)))
        mask = np.array(Image.open(mask_path))[:,:, 0] # take only one channel from 3-channels mask image
        
        
        # Mask clearning (to remove some artifacts coming from labeling)
        if self.mask_clean:
            mask = (mask > 128) # * 255 # if mask value > 128, set it to 255 (this will remove 254, 253 values and 1,2 etc)
        mask = mask.astype("float")
        mask = np.expand_dims(mask, axis=2)
        # extract certain classes from mask (e.g. cars)
        #masks = [(mask == v) for v in self.class_values]
       # mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return {"image":image, "mask":mask, "image_origin": image_origin}
        
    def __len__(self):
        return len(self.images)
    
if __name__=="__main__":
    test_dataset = Dataset(img_dir="/work/vajira/DL/roman_diffusion_model/conditional-polyp-diffusion/Kvasir-SEG/train/images",
                              mask_dir = "/work/vajira/DL/roman_diffusion_model/conditional-polyp-diffusion/Kvasir-SEG/train/masks")
    print(len(test_dataset))
    #img, mask = 
    sample = test_dataset[0]
    print(sample["image"].shape)
    print(sample["mask"].shape)