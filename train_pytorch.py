#!/usr/bin/env python3
"""PyTorch training script for GraphCast weather prediction model."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import xarray as xr
import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging
from pathlib import Path
import argparse
import time
import json

from graphcast import graphcast_torch
from graphcast import xarray_torch
from graphcast.normalization_torch import InputsAndResiduals
from graphcast.autoregressive_torch import Predictor as AutoregressivePredictor


class GraphCastTrainer:
    """PyTorch trainer for GraphCast models."""
    
    def __init__(self, 
                 model_config: graphcast_torch.ModelConfig,
                 task_config: graphcast_torch.TaskConfig,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 mixed_precision: bool = True):
        self.device = torch.device(device)
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler() if mixed_precision else None
        
        self.base_model = graphcast_torch.GraphCast(model_config, task_config)
        self.model = None
        
        self.model_config = model_config
        self.task_config = task_config
        
        logging.info(f"Initialized trainer on device: {self.device}")
        
    def setup_model_with_normalization(self, 
                                     stddev_by_level: xr.Dataset,
                                     mean_by_level: xr.Dataset, 
                                     diffs_stddev_by_level: xr.Dataset):
        """Set up model with normalization wrappers."""
        normalized_model = InputsAndResiduals(
            self.base_model,
            stddev_by_level=stddev_by_level,
            mean_by_level=mean_by_level,
            diffs_stddev_by_level=diffs_stddev_by_level
        )
        
        self.model = AutoregressivePredictor(normalized_model)
        self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
    def compute_loss(self, 
                    inputs: xr.Dataset, 
                    targets: xr.Dataset, 
                    forcings: xr.Dataset) -> Tuple[torch.Tensor, Dict]:
        """Compute loss for training."""
        if self.model is None:
            raise ValueError("Model not set up. Call setup_model_with_normalization first.")
            
        with autocast(enabled=self.mixed_precision):
            loss, diagnostics = self.model.loss(inputs, targets, forcings)
            
        return loss, diagnostics
    
    def train_step(self, 
                  optimizer: optim.Optimizer,
                  inputs: xr.Dataset, 
                  targets: xr.Dataset, 
                  forcings: xr.Dataset) -> Tuple[float, Dict]:
        """Single training step."""
        self.model.train()
        optimizer.zero_grad()
        
        loss, diagnostics = self.compute_loss(inputs, targets, forcings)
        
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        return loss.item(), diagnostics
    
    def validate_step(self,
                     inputs: xr.Dataset,
                     targets: xr.Dataset,
                     forcings: xr.Dataset) -> Tuple[float, Dict]:
        """Single validation step."""
        self.model.eval()
        with torch.no_grad():
            loss, diagnostics = self.compute_loss(inputs, targets, forcings)
        return loss.item(), diagnostics
    
    def save_checkpoint(self, filepath: Path, optimizer: optim.Optimizer, 
                       epoch: int, loss: float):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'model_config': self.model_config,
            'task_config': self.task_config,
        }
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filepath)
        logging.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: Path, optimizer: optim.Optimizer = None):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if self.model is None:
            raise ValueError("Model not set up. Call setup_model_with_normalization first.")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logging.info(f"Checkpoint loaded from {filepath}")
        return checkpoint['epoch'], checkpoint['loss']


def create_synthetic_data(task_config: graphcast_torch.TaskConfig) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Create synthetic weather data for testing."""
    batch_size = 2
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
            shape = (batch_size, time_steps_target, len(levels), lat_size, lon_size)
            dims = ['batch', 'time', 'level', 'lat', 'lon']
        else:
            shape = (batch_size, time_steps_target, lat_size, lon_size)
            dims = ['batch', 'time', 'lat', 'lon']
        
        data = np.random.randn(*shape).astype(np.float32)
        targets[var] = xr.DataArray(data, dims=dims, coords={k: target_coords[k] for k in dims})
    
    forcings = {}
    for var in task_config.forcing_variables:
        shape = (batch_size, time_steps_target, lat_size, lon_size)
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


def main():
    parser = argparse.ArgumentParser(description='Train GraphCast model')
    parser.add_argument('--test-mode', action='store_true', 
                       help='Run in test mode with synthetic data')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    model_config = graphcast_torch.ModelConfig(
        resolution=1,
        mesh_size=2,
        latent_size=64,
        gnn_msg_steps=4,
        hidden_layers=1,
        radius_query_fraction_edge_length=1.0
    )
    
    task_config = graphcast_torch.TaskConfig(
        input_variables=('geopotential', 'temperature', 'u_component_of_wind', 'v_component_of_wind'),
        target_variables=('geopotential', 'temperature'),
        forcing_variables=('2m_temperature',),
        pressure_levels=(500, 850, 1000),
        input_duration='12h'
    )
    
    trainer = GraphCastTrainer(model_config, task_config, device=device)
    
    if args.test_mode:
        logging.info("Running in test mode with synthetic data")
        
        stddev_by_level, mean_by_level, diffs_stddev_by_level = create_synthetic_normalization_stats()
        trainer.setup_model_with_normalization(stddev_by_level, mean_by_level, diffs_stddev_by_level)
        
        optimizer = optim.Adam(trainer.model.parameters(), lr=args.lr)
        
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        for epoch in range(args.epochs):
            logging.info(f"Epoch {epoch + 1}/{args.epochs}")
            
            inputs, targets, forcings = create_synthetic_data(task_config)
            
            start_time = time.time()
            loss, diagnostics = trainer.train_step(optimizer, inputs, targets, forcings)
            step_time = time.time() - start_time
            
            logging.info(f"  Train loss: {loss:.6f}, Step time: {step_time:.3f}s")
            
            if (epoch + 1) % 2 == 0:
                val_loss, val_diagnostics = trainer.validate_step(inputs, targets, forcings)
                logging.info(f"  Val loss: {val_loss:.6f}")
                
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
                trainer.save_checkpoint(checkpoint_path, optimizer, epoch + 1, loss)
        
        logging.info("Training completed successfully!")
        
    else:
        logging.error("Real data training not implemented yet. Use --test-mode for testing.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
