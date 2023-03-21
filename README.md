# divergent-nets-pytorch-lightning


## How to run (Example)
```python
bash run_DeepLabv3plus.sh # This runs multiple config file. Need to edit to run a single config file
```

### Make your own config file with your data paths
Copy one of the confi files and modify it. 
For example: config_RALLFALL.yaml

Don't change the all, but the following parameters:

```python
model:
  arch: UnetPlusPlus
  encoder_name: resnet34
  in_channels: 3
  out_classes: 1
  lr: 0.0001
  test_print_batch_id: 0
  test_print_num: 5
  encoder_weights: imagenet
data:
  encoder: se_resnext50_32x4d
  encoder_weight: imagenet
  bs: 8
  num_workers: 4
  data:
    train:
    - img_dir: "/work/vajira/DL/roman_diffusion_model/conditional-polyp-diffusion/Kvasir-SEG/train/images"
      mask_dir: "/work/vajira/DL/roman_diffusion_model/conditional-polyp-diffusion/Kvasir-SEG/train/masks"
      num_samples: -1
   
    - img_dir: "/work/vajira/DL/roman_diffusion_model/Models/results_135/samples" # fake samples
      mask_dir: "/work/vajira/DL/roman_diffusion_model/Models/200" # input masks are here
      num_samples: -1
    validation:
    - img_dir: "/work/vajira/DL/roman_diffusion_model/conditional-polyp-diffusion/Kvasir-SEG/test/images"
      mask_dir: "/work/vajira/DL/roman_diffusion_model/conditional-polyp-diffusion/Kvasir-SEG/test/masks"
      num_samples: -1
    - img_dir: "/work/vajira/DL/roman_diffusion_model/conditional-polyp-diffusion/Kvasir-SEG/val/images"
      mask_dir: "/work/vajira/DL/roman_diffusion_model/conditional-polyp-diffusion/Kvasir-SEG/val/masks"
      num_samples: -1
wandb_name: 05_config
wandb_entity: simulamet_mlc
wandb_project: diffusion_polyp
output_dir: output/05_config
ckpt_path: null
```

