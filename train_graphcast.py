#!/usr/bin/env python3
"""Comprehensive training script for GraphCast PyTorch model."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import xarray as xr
import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging
from pathlib import Path
import argparse
import time
import json
from tqdm import tqdm

from graphcast import graphcast_torch
from graphcast import xarray_torch
from graphcast.normalization_torch import InputsAndResiduals
from graphcast.autoregressive_torch import Predictor as AutoregressivePredictor
from data_loader_pytorch import SyntheticWeatherDataset, create_weather_dataloader


class GraphCastTrainingManager:
    """Complete training manager for GraphCast models."""
    
    def __init__(self,
                 model_config: graphcast_torch.ModelConfig,
                 task_config: graphcast_torch.TaskConfig,
                 training_config: Dict[str, Any],
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.device = torch.device(device)
        self.model_config = model_config
        self.task_config = task_config
        self.training_config = training_config
        
        self.base_model = graphcast_torch.GraphCast(model_config, task_config)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        if training_config.get('mixed_precision', True):
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        logging.info(f"Training manager initialized on device: {self.device}")
        
    def setup_model(self, 
                   stddev_by_level: xr.Dataset,
                   mean_by_level: xr.Dataset,
                   diffs_stddev_by_level: xr.Dataset):
        """Set up model with normalization and autoregressive wrappers."""
        normalized_model = InputsAndResiduals(
            self.base_model,
            stddev_by_level=stddev_by_level,
            mean_by_level=mean_by_level,
            diffs_stddev_by_level=diffs_stddev_by_level
        )
        
        self.model = AutoregressivePredictor(
            normalized_model,
            noise_level=self.training_config.get('noise_level'),
            gradient_checkpointing=self.training_config.get('gradient_checkpointing', False)
        )
        self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"Model setup complete: {total_params:,} total params, {trainable_params:,} trainable")
        
    def setup_optimizer(self):
        """Set up optimizer and learning rate scheduler."""
        if self.model is None:
            raise ValueError("Model must be set up before optimizer")
        
        optimizer_config = self.training_config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adam')
        lr = optimizer_config.get('lr', 1e-4)
        weight_decay = optimizer_config.get('weight_decay', 0.0)
        
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        elif optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        scheduler_config = self.training_config.get('scheduler', {})
        if scheduler_config.get('type') == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', 100),
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        elif scheduler_config.get('type') == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        
        logging.info(f"Optimizer setup: {optimizer_type} with lr={lr}")
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        epoch_diagnostics = {}
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            inputs = batch['inputs']
            targets = batch['targets']
            forcings = batch['forcings']
            
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                loss, diagnostics = self.model.loss(inputs, targets, forcings)
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                
                if self.training_config.get('gradient_clipping'):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config['gradient_clipping']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                if self.training_config.get('gradient_clipping'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config['gradient_clipping']
                    )
                
                self.optimizer.step()
            
            epoch_losses.append(loss.item())
            
            for key, value in diagnostics.items():
                if key not in epoch_diagnostics:
                    epoch_diagnostics[key] = []
                if hasattr(value, 'item'):
                    epoch_diagnostics[key].append(value.item())
                else:
                    epoch_diagnostics[key].append(float(value))
            
            progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = np.mean(epoch_losses)
        avg_diagnostics = {k: np.mean(v) for k, v in epoch_diagnostics.items()}
        
        return {'loss': avg_loss, **avg_diagnostics}
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_losses = []
        epoch_diagnostics = {}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                inputs = batch['inputs']
                targets = batch['targets']
                forcings = batch['forcings']
                
                loss, diagnostics = self.model.loss(inputs, targets, forcings)
                epoch_losses.append(loss.item())
                
                for key, value in diagnostics.items():
                    if key not in epoch_diagnostics:
                        epoch_diagnostics[key] = []
                    if hasattr(value, 'item'):
                        epoch_diagnostics[key].append(value.item())
                    else:
                        epoch_diagnostics[key].append(float(value))
        
        avg_loss = np.mean(epoch_losses)
        avg_diagnostics = {k: np.mean(v) for k, v in epoch_diagnostics.items()}
        
        return {'loss': avg_loss, **avg_diagnostics}
    
    def save_checkpoint(self, filepath: Path, epoch: int, metrics: Dict):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': self.model_config,
            'task_config': self.task_config,
            'training_config': self.training_config,
            'metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filepath)
        logging.info(f"Checkpoint saved: {filepath}")
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              num_epochs: int = 100,
              checkpoint_dir: Path = Path("checkpoints"),
              save_every: int = 10):
        """Complete training loop."""
        
        checkpoint_dir.mkdir(exist_ok=True)
        
        for epoch in range(num_epochs):
            logging.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics['loss'])
            
            logging.info(f"Train Loss: {train_metrics['loss']:.6f}")
            
            if val_loader is not None:
                val_metrics = self.validate_epoch(val_loader)
                self.val_losses.append(val_metrics['loss'])
                logging.info(f"Val Loss: {val_metrics['loss']:.6f}")
                
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    best_checkpoint = checkpoint_dir / "best_model.pt"
                    self.save_checkpoint(best_checkpoint, epoch + 1, val_metrics)
            
            if self.scheduler:
                self.scheduler.step()
            
            if (epoch + 1) % save_every == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
                metrics = val_metrics if val_loader else train_metrics
                self.save_checkpoint(checkpoint_path, epoch + 1, metrics)
        
        final_checkpoint = checkpoint_dir / "final_model.pt"
        final_metrics = val_metrics if val_loader else train_metrics
        self.save_checkpoint(final_checkpoint, num_epochs, final_metrics)
        
        logging.info("Training completed!")


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
    parser.add_argument('--config', type=str, help='Training configuration file')
    parser.add_argument('--test-mode', action='store_true', 
                       help='Run in test mode with synthetic data')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
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
    
    training_config = {
        'mixed_precision': True,
        'gradient_checkpointing': False,
        'gradient_clipping': 1.0,
        'noise_level': None,
        'optimizer': {
            'type': 'adam',
            'lr': args.lr,
            'weight_decay': 1e-5,
            'betas': (0.9, 0.999)
        },
        'scheduler': {
            'type': 'cosine',
            'T_max': args.epochs,
            'eta_min': 1e-6
        }
    }
    
    trainer = GraphCastTrainingManager(
        model_config, task_config, training_config, device=device
    )
    
    if args.test_mode:
        logging.info("Running in test mode with synthetic data")
        
        stddev_by_level, mean_by_level, diffs_stddev_by_level = create_synthetic_normalization_stats()
        trainer.setup_model(stddev_by_level, mean_by_level, diffs_stddev_by_level)
        trainer.setup_optimizer()
        
        task_config_dict = {
            'input_variables': task_config.input_variables,
            'target_variables': task_config.target_variables,
            'forcing_variables': task_config.forcing_variables,
            'pressure_levels': task_config.pressure_levels
        }
        
        train_dataset = SyntheticWeatherDataset(task_config_dict, num_samples=20)
        val_dataset = SyntheticWeatherDataset(task_config_dict, num_samples=5)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            checkpoint_dir=Path(args.checkpoint_dir),
            save_every=max(1, args.epochs // 5)
        )
        
        logging.info("Training completed successfully!")
        
    else:
        logging.error("Real data training not implemented yet. Use --test-mode for testing.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
