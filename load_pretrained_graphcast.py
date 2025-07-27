#!/usr/bin/env python3
"""Load pretrained GraphCast model from JAX checkpoint into PyTorch."""

import torch
import xarray as xr
from pathlib import Path
from typing import Optional

from graphcast import graphcast_torch
from graphcast.normalization_torch import InputsAndResiduals
from graphcast.autoregressive_torch import Predictor as AutoregressivePredictor
from convert_jax_to_pytorch import (
    convert_jax_checkpoint_to_pytorch,
    create_pytorch_model_from_jax_config,
    load_normalization_stats
)

class PretrainedGraphCast:
    """Wrapper for loading and using pretrained GraphCast models."""
    
    def __init__(self, 
                 jax_checkpoint_path: str,
                 normalization_stats_path: Optional[str] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize pretrained GraphCast model.
        
        Args:
            jax_checkpoint_path: Path to JAX checkpoint file
            normalization_stats_path: Path to normalization statistics
            device: Device to run model on
        """
        self.device = torch.device(device)
        self.jax_checkpoint_path = jax_checkpoint_path
        self.normalization_stats_path = normalization_stats_path
        
        self._load_model()
        
    def _load_model(self):
        """Load the pretrained model."""
        print(f"Loading pretrained GraphCast from {self.jax_checkpoint_path}")
        
        self.base_model = create_pytorch_model_from_jax_config(self.jax_checkpoint_path)
        
        pytorch_checkpoint_path = convert_jax_checkpoint_to_pytorch(
            self.jax_checkpoint_path,
            self.base_model
        )
        
        checkpoint = torch.load(pytorch_checkpoint_path, map_location=self.device)
        self.base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.base_model.to(self.device)
        
        if self.normalization_stats_path:
            norm_stats = load_normalization_stats(self.normalization_stats_path)
            
            self.normalized_model = InputsAndResiduals(
                self.base_model,
                stddev_by_level=norm_stats.get('stddev_by_level'),
                mean_by_level=norm_stats.get('mean_by_level'),
                diffs_stddev_by_level=norm_stats.get('diffs_stddev_by_level')
            )
            
            self.model = AutoregressivePredictor(self.normalized_model)
        else:
            print("Warning: No normalization stats provided, using base model only")
            self.model = self.base_model
        
        self.model.to(self.device)
        print("✓ Pretrained model loaded successfully")
    
    def predict(self, 
                inputs: xr.Dataset,
                targets_template: xr.Dataset,
                forcings: xr.Dataset,
                num_steps: int = 1) -> xr.Dataset:
        """
        Make weather predictions.
        
        Args:
            inputs: Input weather data
            targets_template: Template for output format
            forcings: Forcing variables
            num_steps: Number of prediction steps
            
        Returns:
            Weather predictions
        """
        self.model.eval()
        
        with torch.no_grad():
            if num_steps == 1:
                predictions = self.model(inputs, targets_template, forcings)
            else:
                predictions = self.model(inputs, targets_template, forcings)
        
        return predictions
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_config': self.base_model.model_config,
            'task_config': self.base_model.task_config,
            'device': str(self.device),
            'jax_checkpoint_path': self.jax_checkpoint_path,
            'normalization_stats_path': self.normalization_stats_path
        }

def load_pretrained_graphcast(jax_checkpoint_path: str,
                             normalization_stats_path: Optional[str] = None,
                             device: Optional[str] = None) -> PretrainedGraphCast:
    """
    Convenience function to load pretrained GraphCast model.
    
    Args:
        jax_checkpoint_path: Path to JAX checkpoint
        normalization_stats_path: Path to normalization statistics
        device: Device to run on
        
    Returns:
        Loaded pretrained model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return PretrainedGraphCast(
        jax_checkpoint_path=jax_checkpoint_path,
        normalization_stats_path=normalization_stats_path,
        device=device
    )

def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Load pretrained GraphCast model')
    parser.add_argument('--jax_checkpoint', required=True, help='Path to JAX checkpoint')
    parser.add_argument('--normalization_stats', help='Path to normalization statistics')
    parser.add_argument('--device', default='auto', help='Device to use (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    device = args.device
    if device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model = load_pretrained_graphcast(
            jax_checkpoint_path=args.jax_checkpoint,
            normalization_stats_path=args.normalization_stats,
            device=device
        )
        
        info = model.get_model_info()
        print("\nModel Information:")
        print(f"  Total parameters: {info['total_parameters']:,}")
        print(f"  Trainable parameters: {info['trainable_parameters']:,}")
        print(f"  Device: {info['device']}")
        print(f"  Model config: {info['model_config']}")
        print(f"  Task config: {info['task_config']}")
        
        print("\n✅ Pretrained model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Failed to load pretrained model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
