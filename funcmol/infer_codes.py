import sys
import os
# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置torch._dynamo配置以解决编译兼容性问题
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from tqdm import tqdm
import time
import hydra
from funcmol.utils.utils_nf import load_neural_field
from funcmol.utils.utils_base import setup_fabric
from funcmol.dataset.dataset_field import create_field_loaders, create_gnf_converter
import omegaconf
import torch


@hydra.main(config_path="configs", config_name="infer_codes", version_base=None)
def main(config):
    # 验证并处理 nf_pretrained_path
    nf_pretrained_path = config.get("nf_pretrained_path")
    if not nf_pretrained_path:
        raise ValueError("必须指定 nf_pretrained_path 参数来指定Lightning checkpoint路径")
    
    if not nf_pretrained_path.endswith('.ckpt') or not os.path.exists(nf_pretrained_path):
        raise ValueError(f"指定的checkpoint文件不存在或格式不正确: {nf_pretrained_path}")
    
    # 自动计算 dirname，基于 nf_pretrained_path 的目录
    checkpoint_dir = os.path.dirname(nf_pretrained_path)
    dirname = os.path.join(checkpoint_dir, "codes", config.get("split", "train"))
    
    # 将 config 转换为普通字典以避免结构化配置限制
    if isinstance(config, omegaconf.dictconfig.DictConfig):
        config = omegaconf.OmegaConf.to_container(config, resolve=True)
    
    config["dirname"] = dirname
    
    # initial setup
    fabric = setup_fabric(config)

    # 加载Lightning checkpoint
    # 使用 weights_only=False 以支持包含 omegaconf.DictConfig 的 checkpoint
    checkpoint = torch.load(nf_pretrained_path, map_location='cpu', weights_only=False)
    
    # 从Lightning checkpoint中提取配置
    if 'hyper_parameters' in checkpoint:
        config_model = checkpoint['hyper_parameters']
    else:
        config_model = checkpoint.get('config', {})
    
    # 将 config_model 转换为普通字典以避免结构化配置的限制
    if isinstance(config_model, omegaconf.dictconfig.DictConfig):
        config_model = omegaconf.OmegaConf.to_container(config_model, resolve=True)
    
    for key in config.keys():
        if key in config_model and \
            isinstance(config_model[key], dict) and isinstance(config[key], dict):
            config_model[key].update(config[key])
        else:
            config_model[key] = config[key]
    config = config_model  # update config with checkpoint config
    enc, _ = load_neural_field(checkpoint, fabric, config)

    # 创建GNFConverter实例用于数据加载
    gnf_converter = create_gnf_converter(config)
    
    # data loader
    loader = create_field_loaders(config, gnf_converter, split=config["split"], fabric=fabric)

    # Print config
    fabric.print(f">> config: {config}")
    fabric.print(f">> seed: {config['seed']}")

    # create output directory
    fabric.print(">> saving codes in", config["dirname"])
    os.makedirs(config["dirname"], exist_ok=True)

    # check if codes already exist
    codes_file_path = os.path.join(config["dirname"], "codes.pt")
    if os.path.exists(codes_file_path):
        fabric.print(f">> codes file already exists: {codes_file_path}")
        fabric.print(">> skipping code inference")
        return

    # start eval
    fabric.print(f">> start code inference in {config['split']} split")
    enc.eval()

    with torch.no_grad():
        codes = []
        t0 = time.time()
        for batch in tqdm(loader):
            codes_batch = enc(batch)
            codes.append(codes_batch.detach().cpu())
        
        fabric.print(f">> saving codes to {codes_file_path}")
        codes = torch.cat(codes, dim=0)
        torch.save(codes, codes_file_path)
        del codes

        elapsed_time = time.time() - t0
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        fabric.print(f">> code inference completed in: {int(hours):0>2}h:{int(minutes):0>2}m:{seconds:05.2f}s")


if __name__ == "__main__":
    main()