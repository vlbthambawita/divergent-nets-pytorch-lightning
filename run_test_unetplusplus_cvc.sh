# !/bin/bash

FNOS="000 100 200 300 400 500 600 700 800 900 1000"

for FNO in $FNOS
do
    config_file=test_cvc.yaml
    model_name=config_unetPP_R100F$FNO.yaml
    exp_name=$config_file
    exp_dir=output/$exp_name"_test_cvc"
    ckpt_dir=output/$exp_name
    
    echo ======Running Training - $config_file=======
    #python train.py fit --config=$config_file --trainer.callbacks.init_args.dirpath=$exp_dir --wandb_name=$exp_name --trainer.callbacks.init_args.save_top_k=1
    
    echo ======Running Testing - $exp_name=======
    echo Running Testting - $config_file
    python train.py test --config=$config_file \
                         --trainer.callbacks.init_args.dirpath=$exp_dir \
                         --wandb_name=$exp_name \
                         --ckpt_path=$ckpt_dir/best.ckpt \
                         --output_dir=$exp_dir 
                         
    
    exit
done
  