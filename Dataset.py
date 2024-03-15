import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import Any, Optional

class SignalDataset(Dataset):
    """Dataset to iterate over non-overlapping windows of a signal DataFrame.
    
    Args:
        csv_file (str): Path to the CSV file containing the signal data.
        window_size (int): Number of elements per window.
        n_channels (int): Number of channels in the signal data.
    """
    
    def __init__(self, config) -> None:
        
        self.config      = config
        self.csv_file    = config['csv_file']
        self.df          = pd.read_csv(self.csv_file)
        self.window_size = config['l_win']
        self.n_channels  = config['n_channel']
        
        self.windows = self._create_windows()
    
    def _create_windows(self) -> torch.Tensor:
        signal_data = self.df.values
        num_windows = signal_data.shape[0] // self.window_size
        
        windows = torch.empty((num_windows, self.window_size, self.n_channels))
        for i in range(num_windows):
            start = i * self.window_size
            end = start + self.window_size
            windows[i] = torch.tensor(signal_data[start:end]).float()
        
        return windows
    
    def __getitem__(self, index: int) -> torch.Tensor:
        return self.windows[index]
    
    def __len__(self) -> int:
        return self.windows.shape[0]

def prepare_dataloader(dataset: SignalDataset, batch_size: int, is_distributed: bool = False, **kwargs: Any) -> DataLoader:
    """Creates a DataLoader for training.
    
    Args:
        dataset (SignalDataset): Training dataset.
        batch_size (int): DataLoader batch size.
        is_distributed (bool): Is the training distributed over multiple nodes? Defaults to False.
    
    Returns:
        DataLoader: Data iterator ready to use.
    """
    sampler: Optional[DistributedSampler] = DistributedSampler(dataset) if is_distributed else None
    
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True, **kwargs)
