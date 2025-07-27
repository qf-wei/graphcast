#!/usr/bin/env python3
"""PyTorch data loading utilities for GraphCast weather data."""

import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
from typing import Tuple, Dict, Optional, List, Callable
from pathlib import Path

from graphcast import xarray_torch


class WeatherDataset(Dataset):
    """PyTorch Dataset for weather prediction data."""
    
    def __init__(self,
                 data_path: str,
                 task_config: Dict,
                 input_duration: str = "12h",
                 target_lead_times: str = "12h",
                 transform: Optional[Callable] = None):
        self.data_path = Path(data_path)
        self.task_config = task_config
        self.input_duration = input_duration
        self.target_lead_times = target_lead_times
        self.transform = transform
        
        self.dataset = self._load_data()
        self.samples = self._create_samples()
        
    def _load_data(self) -> xr.Dataset:
        """Load weather data from file."""
        if self.data_path.suffix == '.nc':
            dataset = xr.open_dataset(self.data_path)
        elif self.data_path.suffix == '.zarr':
            dataset = xr.open_zarr(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
            
            
        return dataset
    
    def _create_samples(self) -> List[Dict]:
        """Create list of training samples."""
        samples = []
        
        inputs = self.dataset[list(self.task_config['input_variables'])]
        targets = self.dataset[list(self.task_config['target_variables'])]
        forcings = self.dataset[list(self.task_config['forcing_variables'])]
        
        samples.append({
            'inputs': inputs,
            'targets': targets,
            'forcings': forcings
        })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, xr.Dataset]:
        sample = self.samples[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


def create_weather_dataloader(data_path: str,
                            task_config: Dict,
                            batch_size: int = 1,
                            shuffle: bool = True,
                            num_workers: int = 0,
                            **kwargs) -> DataLoader:
    """Create DataLoader for weather data."""
    dataset = WeatherDataset(data_path, task_config, **kwargs)
    
    def collate_fn(batch):
        """Custom collate function for xarray datasets."""
        if len(batch) == 1:
            return batch[0]
        
        inputs_list = [sample['inputs'] for sample in batch]
        targets_list = [sample['targets'] for sample in batch]
        forcings_list = [sample['forcings'] for sample in batch]
        
        inputs = xr.concat(inputs_list, dim='batch')
        targets = xr.concat(targets_list, dim='batch')
        forcings = xr.concat(forcings_list, dim='batch')
        
        return {
            'inputs': inputs,
            'targets': targets,
            'forcings': forcings
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


class SyntheticWeatherDataset(Dataset):
    """Synthetic weather dataset for testing."""
    
    def __init__(self,
                 task_config: Dict,
                 num_samples: int = 100,
                 input_duration: str = "12h",
                 target_lead_times: str = "12h"):
        self.task_config = task_config
        self.num_samples = num_samples
        self.input_duration = input_duration
        self.target_lead_times = target_lead_times
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, xr.Dataset]:
        """Generate synthetic weather data sample."""
        batch_size = 1
        time_steps_input = 2
        time_steps_target = 1
        lat_size = 32
        lon_size = 64
        levels = [500, 850, 1000]
        
        time_input = np.arange(time_steps_input) * np.timedelta64(6, 'h')
        time_target = np.arange(time_steps_target) * np.timedelta64(6, 'h') + time_input[-1] + np.timedelta64(6, 'h')
        
        coords = {
            'batch': np.arange(batch_size),
            'time': time_input,
            'lat': np.linspace(-90, 90, lat_size),
            'lon': np.linspace(0, 360, lon_size, endpoint=False),
            'level': levels
        }
        
        target_coords = coords.copy()
        target_coords['time'] = time_target
        
        np.random.seed(idx)
        
        inputs = {}
        for var in self.task_config['input_variables']:
            if var in ['geopotential', 'temperature', 'u_component_of_wind', 'v_component_of_wind']:
                shape = (batch_size, time_steps_input, len(levels), lat_size, lon_size)
                dims = ['batch', 'time', 'level', 'lat', 'lon']
            else:
                shape = (batch_size, time_steps_input, lat_size, lon_size)
                dims = ['batch', 'time', 'lat', 'lon']
            
            data = np.random.randn(*shape).astype(np.float32)
            inputs[var] = xr.DataArray(data, dims=dims, coords={k: coords[k] for k in dims})
        
        targets = {}
        for var in self.task_config['target_variables']:
            if var in ['geopotential', 'temperature', 'u_component_of_wind', 'v_component_of_wind']:
                shape = (batch_size, time_steps_target, len(levels), lat_size, lon_size)
                dims = ['batch', 'time', 'level', 'lat', 'lon']
            else:
                shape = (batch_size, time_steps_target, lat_size, lon_size)
                dims = ['batch', 'time', 'lat', 'lon']
            
            data = np.random.randn(*shape).astype(np.float32)
            targets[var] = xr.DataArray(data, dims=dims, coords={k: target_coords[k] for k in dims})
        
        forcings = {}
        for var in self.task_config['forcing_variables']:
            shape = (batch_size, time_steps_target, lat_size, lon_size)
            dims = ['batch', 'time', 'lat', 'lon']
            data = np.random.randn(*shape).astype(np.float32)
            forcings[var] = xr.DataArray(data, dims=dims, coords={k: target_coords[k] for k in dims})
        
        return {
            'inputs': xr.Dataset(inputs),
            'targets': xr.Dataset(targets),
            'forcings': xr.Dataset(forcings)
        }
