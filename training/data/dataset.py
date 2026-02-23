import logging
from typing import Union

from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

tqdm.pandas()


class CIFAR10(Dataset):
    def __init__(
        self
    ) -> None:
            pass
    def __len__(self):
        pass

    def __getitem__(self, idx: Union[int, slice]):
            pass
    
