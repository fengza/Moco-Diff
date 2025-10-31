from typing import Mapping, Any
import importlib
import os
from torch import nn

curPath_ = os.path.dirname(__file__)
curPath = os.getcwd()
def get_obj_from_str(string: str, reload: bool=False) -> object:
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: Mapping[str, Any]) -> object:
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def disabled_train(self: nn.Module) -> nn.Module:
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def frozen_module(module: nn.Module) -> None:
    module.eval()
    module.train = disabled_train
    for p in module.parameters():
        p.requires_grad = False


def load_state_dict(model: nn.Module, state_dict: Mapping[str, Any], strict: bool=False) -> None:
    
    # with open('/home_data/home/lifeng2023/code/moco/DiffBIR-main/checkpoints/model.txt','a') as f:
	# # 打印模型到model.txt
    #     print(model, file = f)
    #     # 打印模型参数
    #     for params in model.state_dict():   
    #         f.write("{}\t{}\n".format(params, model.state_dict()[params]))
    state_dict = state_dict.get("state_dict", state_dict)
    #state_dict.pop('conv_first.1.weight')
    is_model_key_starts_with_module = list(model.state_dict().keys())[0].startswith("module.")
    is_state_dict_key_starts_with_module = list(state_dict.keys())[0].startswith("module.")
    
    if (
        is_model_key_starts_with_module and
        (not is_state_dict_key_starts_with_module)
    ):
        state_dict = {f"module.{key}": value for key, value in state_dict.items()}
    if (
        (not is_model_key_starts_with_module) and
        is_state_dict_key_starts_with_module
    ):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=strict)
