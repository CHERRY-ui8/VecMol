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
from funcmol.utils.utils_nf import load_neural_field, get_latest_model_path
from funcmol.utils.utils_base import setup_fabric
from funcmol.dataset.dataset_field import create_field_loaders, create_gnf_converter
import omegaconf
import torch


@hydra.main(config_path="configs", config_name="infer_codes", version_base=None)
def main(config):
    # 自动获取最新的模型路径
    if "nf_pretrained_path" not in config or config["nf_pretrained_path"] is None:
        config["nf_pretrained_path"] = get_latest_model_path()
    
    # initial setup
    fabric = setup_fabric(config)

    # load checkpoint and update config
    checkpoint = fabric.load(os.path.join(config["nf_pretrained_path"], "model.pt"))
    config_model = checkpoint["config"]
    for key in config.keys():
        if key in config_model and \
            isinstance(config_model[key], omegaconf.dictconfig.DictConfig):
            config_model[key].update(config[key])
        else:
            config_model[key] = config[key]
    config = config_model  # update config with checkpoint config
    enc, _ = load_neural_field(checkpoint, fabric, config)

    # 创建GNFConverter实例用于数据加载
    gnf_converter = create_gnf_converter(config, device="cpu")
    
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