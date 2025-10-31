from argparse import ArgumentParser
import os
import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch
from utils.common import frozen_module
from pytorch_lightning.plugins import DDPPlugin
#from pytorch_lightning.strategies.ddp import DDPStrategy

from utils.common import instantiate_from_config, load_state_dict
curPath_ = os.path.dirname(__file__)
curPath = os.getcwd()
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default='/public_bme/data/lifeng/code/moco/TS_BHIR/configs/train_class.yaml', required=False) #'/public_bme/data/lifeng/code/moco/DiffBIR-main/configs/train_cldm.yaml'
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config)
    pl.seed_everything(config.lightning.seed, workers=True)
    
    data_module = instantiate_from_config(config.data)
    model = instantiate_from_config(OmegaConf.load(config.model.config))
    #model_dict = model.state_dict()
    #将要输出保存的文件地址，若文件不存在，则会自动创建
    # with open('/public_bme/data/lifeng/code/moco/DiffBIR-main/full_model_arch.txt','a') as file0:
    #     print(model, file=file0)
    
    # TODO: resume states saved in checkpoint.
    if config.model.get("resume"):
        #pretrained_dict = torch.load(config.model.resume)['state_dict']
        #pretrained_dict = pretrained_dict.state_dict
        # 1. filter out unnecessary keys
        #pretrained_dict = {k: v for k, v in model_dict.items() if k in pretrained_dict}
        # 2. overwrite entries in the existing state dict
        #model_dict.update(pretrained_dict)
        # 3. load the new state dict
        #model.load_state_dict(model_dict)
        load_state_dict(model, torch.load(config.model.resume), strict=False) #, strict=True, map_location="cpu"

    if config.model.get("quality_resume"):
        load_state_dict(model.quality_predictor, torch.load(config.model.quality_resume, map_location="cpu"), strict=True)
    #frozen_module(model.quality_predictor)
    if config.model.get('network_1_resume'):
        load_state_dict(model.network_1, torch.load(config.model.network_1_resume, map_location="cpu"), strict=True)
    if config.model.get('network_2_resume'):
        load_state_dict(model.network_2, torch.load(config.model.network_2_resume, map_location="cpu"), strict=True)

    frozen_module(model)
    #for param in model.parameters():
    for name, param in model.quality_predictor.named_parameters():
    #         #if name == 'conv_first_revised' or name == 'conv_last' or name == 'conv_up2' or name == 'conv_before_upsample' or name == 'conv_up1' or name == 'conv_hr' or name == 'lrelu':
    #         if name.startswith('layers'):# or name.startswith('fusion_block'):
    #             param.requires_grad = False
        #if name.startswith("img_router"):
        if name.startswith("quality_control"):
            param.requires_grad = True
            
    callbacks = []
    for callback_config in config.lightning.callbacks:
        callbacks.append(instantiate_from_config(callback_config))
    trainer = pl.Trainer(callbacks=callbacks, **config.lightning.trainer)  #,plugins=DDPPlugin(find_unused_parameters=True)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
