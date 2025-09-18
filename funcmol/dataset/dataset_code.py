import os
import random

import torch
from torch.utils.data import Dataset, Subset


class CodeDataset(Dataset):
    """
    A dataset class for handling code files in a specified directory.

    Attributes:
        dset_name (str): The name of the dataset. Default is "qm9".
        split (str): The data split to use (e.g., "train", "test"). Default is "train".
        codes_dir (str): The directory where the code files are stored.
        num_augmentations (int): The number of augmentations to use. If None and split is "train", it defaults to the number of code files minus one.

    Methods:
        __len__(): Returns the number of samples in the current codes.
        __getitem__(index): Returns the code at the specified index.
        load_codes(index=None): Loads the codes from the specified index or a random index if None.
    """
    def __init__(
        self,
        dset_name: str = "qm9",
        split: str = "train",
        codes_dir: str = None,
        num_augmentations = None,  # 保留参数以兼容现有代码，但不再使用
    ):
        self.dset_name = dset_name
        self.split = split
        self.codes_dir = os.path.join(codes_dir, self.split)

        # get list of codes
        self.list_codes = [
            f for f in os.listdir(self.codes_dir)
            if os.path.isfile(os.path.join(self.codes_dir, f)) and \
            f.startswith("codes") and f.endswith(".pt")
        ]
        # 简化：不再使用数据增强，直接加载codes.pt
        if "codes.pt" in self.list_codes:
            self.list_codes = ["codes.pt"]
        else:
            # 兼容旧格式：如果有编号的codes文件，使用第一个
            self.list_codes.sort()
            if self.list_codes:
                self.list_codes = [self.list_codes[0]]
        self.num_augmentations = 0  # 不再使用数据增强
        self.load_codes(0)

    def __len__(self):
        return self.curr_codes.shape[0]

    def __getitem__(self, index):
        return self.curr_codes[index]

    def load_codes(self, index=None) -> None:
        """
        Load codes from the available codes file.

        Args:
            index (int, optional): The index of the code to load. If None, uses the first (and only) file.

        Returns:
            None

        Side Effects:
            - Sets `self.curr_codes` to the loaded codes from the codes file.
            - Prints the path of the loaded codes.
        """
        if index is None:
            index = 0
        code_path = os.path.join(self.codes_dir, self.list_codes[index])
        print(">> loading codes: ", code_path)
        self.curr_codes = torch.load(code_path, weights_only=False)


def create_code_loaders(
    config: dict,
    split: str = None,
    # fabric = None,
):
    """
    Creates and returns a DataLoader for the specified dataset split.

    Args:
        config (dict): Configuration dictionary containing dataset parameters.
            - dset (dict): Dictionary with dataset-specific parameters.
                - dset_name (str): Name of the dataset.
            - codes_dir (str): Directory where the code files are stored.
            - num_augmentations (int): Number of augmentations to apply to the dataset.
            - debug (bool): Flag to indicate if debugging mode is enabled.
            - dset (dict): Dictionary with DataLoader parameters.
                - batch_size (int): Batch size for the DataLoader.
                - num_workers (int): Number of worker threads for data loading.
        split (str, optional): The dataset split to load (e.g., 'train', 'val', 'test'). Defaults to None.
        fabric: Fabric object for setting up the DataLoader.

    Returns:
        DataLoader: A PyTorch DataLoader object for the specified dataset split.
    """
    dset = CodeDataset(
        dset_name=config["dset"]["dset_name"],
        codes_dir=config["codes_dir"],
        split=split,
        num_augmentations=config["num_augmentations"],
    )

    # reduce the dataset size for debugging
    if config["debug"] or split in ["val", "test"]:
        indexes = list(range(len(dset)))
        random.Random(0).shuffle(indexes)
        indexes = indexes[:5000]
        if len(dset) > len(indexes):
            dset = Subset(dset, indexes)  # Smaller training set for debugging

    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=min(config["dset"]["batch_size"], len(dset)),
        num_workers=config["dset"]["num_workers"],
        shuffle=True if split == "train" else False,
        pin_memory=True,
        drop_last=True,
    )
    # fabric.print(f">> {split} set size: {len(dset)}")
    print(f">> {split} set size: {len(dset)}")

    return loader # fabric.setup_dataloaders(loader, use_distributed_sampler=(split == "train"))
