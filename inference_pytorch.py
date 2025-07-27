#!/usr/bin/env python3
"""PyTorch inference script for GraphCast weather prediction model."""

import torch
import torch.nn as nn
import xarray as xr
import numpy as np
from typing import Dict, List, Optional, Iterator, Tuple
import logging
from pathlib import Path
import argparse
import time

from graphcast import graphcast_torch
from graphcast import xarray_torch
from graphcast.rollout_torch import chunked_prediction_generator_multiple_runs, chunked_prediction_generator


class GraphCastInference:
    """PyTorch inference engine for GraphCast models."""
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        
        logging.info(f"Initialized inference engine on device: {self.device}")
        
    def predict_single_step(self,
                           inputs: xr.Dataset,
                           targets_template: xr.Dataset,
                           forcings: xr.Dataset,
                           **kwargs) -> xr.Dataset:
        """Single step prediction."""
        with torch.no_grad():
            predictions = self.model(inputs, targets_template, forcings, **kwargs)
        return predictions
    
    def predict_autoregressive(self,
                              inputs: xr.Dataset,
                              targets_template: xr.Dataset,
                              forcings: xr.Dataset,
                              **kwargs) -> xr.Dataset:
        """Multi-step autoregressive prediction."""
        with torch.no_grad():
            predictions = self.model(inputs, targets_template, forcings, **kwargs)
        return predictions
    
    def predict_ensemble(self,
                        inputs: xr.Dataset,
                        targets_template: xr.Dataset,
                        forcings: xr.Dataset,
                        num_ensemble_members: int = 8,
                        seed: int = 42,
                        **kwargs) -> xr.Dataset:
        """Ensemble prediction with multiple random seeds."""
        torch.manual_seed(seed)
        
        ensemble_predictions = []
        for i in range(num_ensemble_members):
            torch.manual_seed(seed + i)
            
            with torch.no_grad():
                pred = self.model(inputs, targets_template, forcings, **kwargs)
            ensemble_predictions.append(pred.expand_dims('sample'))
        
        return xr.concat(ensemble_predictions, dim='sample')
    
    def chunked_prediction(self,
                          inputs: xr.Dataset,
                          targets_template: xr.Dataset,
                          forcings: xr.Dataset,
                          num_steps_per_chunk: int = 1,
                          **kwargs) -> Iterator[xr.Dataset]:
        """Memory-efficient chunked prediction."""
        for chunk in chunked_prediction_generator(
            predictor_fn=self.predict_single_step,
            inputs=inputs,
            targets_template=targets_template,
            forcings=forcings,
            num_steps_per_chunk=num_steps_per_chunk,
            **kwargs
        ):
            yield chunk
    
    def predict_with_rollout(self,
                           inputs: xr.Dataset,
                           targets_template: xr.Dataset,
                           forcings: xr.Dataset,
                           num_ensemble_members: int = 1,
                           num_steps_per_chunk: int = 1,
                           **kwargs) -> xr.Dataset:
        """Prediction with ensemble rollout."""
        chunks = []
        for chunk in chunked_prediction_generator_multiple_runs(
            predictor_fn=self.predict_single_step,
            inputs=inputs,
            targets_template=targets_template,
            forcings=forcings,
            num_steps_per_chunk=num_steps_per_chunk,
            num_samples=num_ensemble_members,
            **kwargs
        ):
            chunks.append(chunk)
        
        if chunks:
            return xr.combine_by_coords(chunks)
        else:
            return xr.Dataset()
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: Path, device: str = "auto"):
        """Load model from checkpoint."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model_config = checkpoint['model_config']
        task_config = checkpoint['task_config']
        
        from graphcast.normalization_torch import InputsAndResiduals
        from graphcast.autoregressive_torch import Predictor as AutoregressivePredictor
        
        base_model = graphcast_torch.GraphCast(model_config, task_config)
        
        stddev_by_level, mean_by_level, diffs_stddev_by_level = create_synthetic_normalization_stats()
        normalized_model = InputsAndResiduals(
            base_model,
            stddev_by_level=stddev_by_level,
            mean_by_level=mean_by_level,
            diffs_stddev_by_level=diffs_stddev_by_level
        )
        model = AutoregressivePredictor(normalized_model)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return cls(model, device=device)


def create_synthetic_data(task_config: graphcast_torch.TaskConfig, 
                         num_target_steps: int = 4) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Create synthetic weather data for testing."""
    batch_size = 1
    time_steps_input = 2
    lat_size = 32
    lon_size = 64
    levels = [500, 850, 1000]
    
    time_input = np.arange(time_steps_input) * np.timedelta64(6, 'h')
    time_target = np.arange(num_target_steps) * np.timedelta64(6, 'h') + time_input[-1] + np.timedelta64(6, 'h')
    
    coords = {
        'batch': np.arange(batch_size),
        'time': time_input,
        'lat': np.linspace(-90, 90, lat_size),
        'lon': np.linspace(0, 360, lon_size, endpoint=False),
        'level': levels
    }
    
    target_coords = coords.copy()
    target_coords['time'] = time_target
    
    inputs = {}
    for var in task_config.input_variables:
        if var in ['geopotential', 'temperature', 'u_component_of_wind', 'v_component_of_wind']:
            shape = (batch_size, time_steps_input, len(levels), lat_size, lon_size)
            dims = ['batch', 'time', 'level', 'lat', 'lon']
        else:
            shape = (batch_size, time_steps_input, lat_size, lon_size)
            dims = ['batch', 'time', 'lat', 'lon']
        
        data = np.random.randn(*shape).astype(np.float32)
        inputs[var] = xr.DataArray(data, dims=dims, coords={k: coords[k] for k in dims})
    
    targets = {}
    for var in task_config.target_variables:
        if var in ['geopotential', 'temperature', 'u_component_of_wind', 'v_component_of_wind']:
            shape = (batch_size, num_target_steps, len(levels), lat_size, lon_size)
            dims = ['batch', 'time', 'level', 'lat', 'lon']
        else:
            shape = (batch_size, num_target_steps, lat_size, lon_size)
            dims = ['batch', 'time', 'lat', 'lon']
        
        data = np.random.randn(*shape).astype(np.float32)
        targets[var] = xr.DataArray(data, dims=dims, coords={k: target_coords[k] for k in dims})
    
    forcings = {}
    for var in task_config.forcing_variables:
        shape = (batch_size, num_target_steps, lat_size, lon_size)
        dims = ['batch', 'time', 'lat', 'lon']
        data = np.random.randn(*shape).astype(np.float32)
        forcings[var] = xr.DataArray(data, dims=dims, coords={k: target_coords[k] for k in dims})
    
    return xr.Dataset(inputs), xr.Dataset(targets), xr.Dataset(forcings)


def create_synthetic_normalization_stats() -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Create synthetic normalization statistics."""
    levels = [500, 850, 1000]
    variables = ['geopotential', 'temperature', 'u_component_of_wind', 'v_component_of_wind', 
                 'mean_sea_level_pressure', '2m_temperature']
    
    coords = {'level': levels}
    
    stddev_data = {}
    mean_data = {}
    diffs_stddev_data = {}
    
    for var in variables:
        if var in ['geopotential', 'temperature', 'u_component_of_wind', 'v_component_of_wind']:
            dims = ['level']
            shape = (len(levels),)
        else:
            dims = []
            shape = ()
        
        stddev_data[var] = xr.DataArray(
            np.ones(shape) + np.random.rand(*shape) * 0.1,
            dims=dims,
            coords={k: coords[k] for k in dims} if dims else {}
        )
        mean_data[var] = xr.DataArray(
            np.random.randn(*shape) * 0.1,
            dims=dims,
            coords={k: coords[k] for k in dims} if dims else {}
        )
        diffs_stddev_data[var] = xr.DataArray(
            np.ones(shape) * 0.5 + np.random.rand(*shape) * 0.1,
            dims=dims,
            coords={k: coords[k] for k in dims} if dims else {}
        )
    
    return (xr.Dataset(stddev_data), 
            xr.Dataset(mean_data), 
            xr.Dataset(diffs_stddev_data))
