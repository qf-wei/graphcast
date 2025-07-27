#!/usr/bin/env python3
"""Example usage of JAX to PyTorch parameter conversion."""

import torch
import numpy as np
from pathlib import Path

from convert_jax_to_pytorch import (
    convert_jax_checkpoint_to_pytorch,
    create_pytorch_model_from_jax_config,
    load_normalization_stats
)
from load_pretrained_graphcast import load_pretrained_graphcast

def example_conversion():
    """Example of converting JAX checkpoint to PyTorch."""
    
    jax_checkpoint_path = "path/to/jax_checkpoint.npz"
    
    if Path(jax_checkpoint_path).exists():
        print("Converting JAX checkpoint to PyTorch...")
        
        pytorch_model = create_pytorch_model_from_jax_config(jax_checkpoint_path)
        
        pytorch_checkpoint_path = convert_jax_checkpoint_to_pytorch(
            jax_checkpoint_path,
            pytorch_model,
            "converted_pytorch_model.pth"
        )
        
        print(f"✓ Conversion completed: {pytorch_checkpoint_path}")
        
        norm_stats_path = "path/to/normalization_stats/"
        if Path(norm_stats_path).exists():
            norm_stats = load_normalization_stats(norm_stats_path)
            print(f"✓ Loaded normalization stats: {list(norm_stats.keys())}")
    
    try:
        pretrained_model = load_pretrained_graphcast(
            jax_checkpoint_path="path/to/jax_checkpoint.npz",
            normalization_stats_path="path/to/normalization_stats/",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        model_info = pretrained_model.get_model_info()
        print(f"✓ Loaded pretrained model with {model_info['total_parameters']:,} parameters")
        
    except Exception as e:
        print(f"Could not load pretrained model: {e}")

def create_mock_checkpoint_for_testing():
    """Create a mock JAX checkpoint for testing purposes."""
    
    mock_data = {
        'params': {
            'grid2mesh_gnn': {
                'processor_networks': {
                    '0': {
                        'update_edge_fn': {
                            'grid2mesh': {
                                'mlp': {
                                    'layers': {
                                        '0': {
                                            'w': np.random.randn(64, 256).astype(np.float32),
                                            'b': np.random.randn(64).astype(np.float32)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        'model_config': {
            'resolution': 0,
            'mesh_size': 6,
            'latent_size': 64,
            'gnn_msg_steps': 2,
            'hidden_layers': 1,
            'radius_query_fraction_edge_length': 0.6
        },
        'task_config': {
            'input_variables': ('geopotential', 'temperature'),
            'target_variables': ('geopotential', 'temperature'),
            'forcing_variables': ('mean_sea_level_pressure',),
            'pressure_levels': (500, 700, 850),
            'input_duration': '12h'
        }
    }
    
    np.savez('mock_jax_checkpoint.npz', **mock_data)
    print("✓ Created mock JAX checkpoint: mock_jax_checkpoint.npz")

if __name__ == "__main__":
    print("JAX to PyTorch Conversion Example")
    print("=" * 40)
    
    create_mock_checkpoint_for_testing()
    
    example_conversion()
