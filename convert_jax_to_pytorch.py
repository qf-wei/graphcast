#!/usr/bin/env python3
"""Convert JAX GraphCast checkpoints to PyTorch format."""

import torch
import numpy as np
import pickle
import xarray as xr
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from graphcast import graphcast_torch
from graphcast import checkpoint
from graphcast import xarray_torch

def load_jax_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load JAX checkpoint file."""
    print(f"Loading JAX checkpoint from {checkpoint_path}")
    
    if checkpoint_path.endswith('.npz'):
        data = np.load(checkpoint_path, allow_pickle=True)
        checkpoint_data = {key: data[key] for key in data.files}
    else:
        checkpoint_data = checkpoint.load(checkpoint_path)
    
    return checkpoint_data

def convert_jax_array_to_torch(jax_array) -> torch.Tensor:
    """Convert JAX array to PyTorch tensor."""
    if hasattr(jax_array, '__array__'):
        numpy_array = np.array(jax_array)
    else:
        numpy_array = jax_array
    
    return torch.from_numpy(numpy_array)

def map_jax_params_to_pytorch(jax_params: Dict[str, Any], 
                             pytorch_model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Map JAX parameter structure to PyTorch state_dict format."""
    
    _trigger_pytorch_parameter_creation(pytorch_model)
    
    pytorch_state_dict = {}
    pytorch_keys = set(pytorch_model.state_dict().keys())
    
    print("JAX parameter structure:")
    print_nested_dict_structure(jax_params)
    
    print(f"\nPyTorch state_dict structure ({len(pytorch_keys)} parameters):")
    for key in sorted(list(pytorch_keys)[:10]):  # Show first 10
        print(f"  {key}: {pytorch_model.state_dict()[key].shape}")
    if len(pytorch_keys) > 10:
        print(f"  ... and {len(pytorch_keys) - 10} more parameters")
    
    def find_pytorch_parameter(jax_path: str, tensor_shape: tuple) -> Optional[str]:
        """Find matching PyTorch parameter by path and shape."""
        jax_components = jax_path.split('.')
        
        pytorch_candidates = []
        
        if 'grid2mesh_gnn' in jax_path:
            pytorch_path = jax_path.replace('grid2mesh_gnn', '_grid2mesh_gnn')
        elif 'mesh_gnn' in jax_path:
            pytorch_path = jax_path.replace('mesh_gnn', '_mesh_gnn')
        elif 'mesh2grid_gnn' in jax_path:
            pytorch_path = jax_path.replace('mesh2grid_gnn', '_mesh2grid_gnn')
        else:
            pytorch_path = jax_path
            
        pytorch_path = pytorch_path.replace('.w', '.weight').replace('.b', '.bias')
        pytorch_path = pytorch_path.replace('.scale', '.weight').replace('.offset', '.bias')
        
        if pytorch_path in pytorch_keys:
            return pytorch_path
            
        for pytorch_key in pytorch_keys:
            pytorch_tensor = pytorch_model.state_dict()[pytorch_key]
            if pytorch_tensor.shape == tensor_shape:
                if any(component in pytorch_key for component in jax_components):
                    return pytorch_key
        
        return None
    
    def convert_nested_params(jax_dict, path_prefix: str = ""):
        """Recursively convert nested parameter dictionaries."""
        if isinstance(jax_dict, np.ndarray):
            print(f"⚠ Warning: Expected dictionary but got numpy array at {path_prefix or 'root'}")
            return
        elif not isinstance(jax_dict, dict):
            print(f"⚠ Warning: Expected dictionary but got {type(jax_dict)} at {path_prefix or 'root'}")
            return
            
        for jax_key, jax_value in jax_dict.items():
            current_path = f"{path_prefix}.{jax_key}" if path_prefix else jax_key
            
            if isinstance(jax_value, dict):
                convert_nested_params(jax_value, current_path)
            else:
                try:
                    tensor = convert_jax_array_to_torch(jax_value)
                    pytorch_key = find_pytorch_parameter(current_path, tensor.shape)
                    
                    if pytorch_key:
                        pytorch_state_dict[pytorch_key] = tensor
                        print(f"✓ Mapped {current_path} -> {pytorch_key}: {tensor.shape}")
                    else:
                        print(f"⚠ No PyTorch parameter found for {current_path} with shape {tensor.shape}")
                        
                except Exception as e:
                    print(f"❌ Error converting parameter {current_path}: {e}")
    
    convert_nested_params(jax_params)
    print(f"\nSuccessfully mapped {len(pytorch_state_dict)}/{len(pytorch_keys)} parameters")
    
    return pytorch_state_dict

def _trigger_pytorch_parameter_creation(pytorch_model: torch.nn.Module):
    """Trigger parameter creation in PyTorch model via forward pass."""
    try:
        batch_size = 2
        time_steps = 2
        levels = 3
        lat_size = 8
        lon_size = 16
        
        coords = {
            'batch': np.arange(batch_size),
            'time': np.arange(time_steps),
            'level': np.array([500, 700, 850]),
            'lat': np.linspace(-90, 90, lat_size),
            'lon': np.linspace(0, 360, lon_size, endpoint=False)
        }
        
        inputs = xarray_torch.Dataset({
            var_name: xarray_torch.DataArray(
                torch.randn(batch_size, time_steps, levels, lat_size, lon_size),
                dims=['batch', 'time', 'level', 'lat', 'lon'],
                coords=coords
            ) for var_name in pytorch_model.task_config.input_variables
        })
        
        targets = xarray_torch.Dataset({
            var_name: xarray_torch.DataArray(
                torch.randn(batch_size, time_steps, levels, lat_size, lon_size),
                dims=['batch', 'time', 'level', 'lat', 'lon'],
                coords=coords
            ) for var_name in pytorch_model.task_config.target_variables
        })
        
        forcings = xarray_torch.Dataset({
            var_name: xarray_torch.DataArray(
                torch.randn(batch_size, time_steps, lat_size, lon_size),
                dims=['batch', 'time', 'lat', 'lon'],
                coords={k: v for k, v in coords.items() if k != 'level'}
            ) for var_name in pytorch_model.task_config.forcing_variables
        })
        
        with torch.no_grad():
            _ = pytorch_model(inputs, targets, forcings)
            
        print(f"✓ Triggered parameter creation: {sum(p.numel() for p in pytorch_model.parameters()):,} parameters")
        
    except Exception as e:
        print(f"⚠ Warning: Could not trigger parameter creation: {e}")

def print_nested_dict_structure(d: Dict, indent: int = 0):
    """Print the structure of a nested dictionary."""
    if isinstance(d, np.ndarray):
        print("  " * indent + f"numpy array: {d.shape}")
        return
    elif not isinstance(d, dict):
        print("  " * indent + f"non-dict: {type(d)}")
        return
        
    for key, value in d.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_nested_dict_structure(value, indent + 1)
        else:
            shape = getattr(value, 'shape', 'unknown')
            print("  " * indent + f"{key}: {shape}")

def convert_jax_checkpoint_to_pytorch(jax_checkpoint_path: str,
                                    pytorch_model: torch.nn.Module,
                                    output_path: Optional[str] = None) -> str:
    """Convert complete JAX checkpoint to PyTorch format."""
    
    jax_data = load_jax_checkpoint(jax_checkpoint_path)
    
    if 'params' in jax_data:
        jax_params = jax_data['params']
        if isinstance(jax_params, np.ndarray):
            try:
                jax_params = jax_params.item()  # Extract from 0-d array
                if not isinstance(jax_params, dict):
                    print(f"Warning: params extracted from numpy array is not a dict: {type(jax_params)}")
                    jax_params = {}
            except:
                print("Warning: Could not extract params from numpy array, using empty dict")
                jax_params = {}
    else:
        jax_params = jax_data
    
    pytorch_state_dict = map_jax_params_to_pytorch(jax_params, pytorch_model)
    
    missing_keys, unexpected_keys = pytorch_model.load_state_dict(pytorch_state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys in PyTorch model: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in conversion: {unexpected_keys}")
    
    if output_path is None:
        output_path = jax_checkpoint_path.replace('.npz', '_pytorch.pth')
    
    torch.save({
        'model_state_dict': pytorch_model.state_dict(),
        'jax_params': jax_params,  # Keep original for reference
        'conversion_info': {
            'missing_keys': missing_keys,
            'unexpected_keys': unexpected_keys,
            'original_checkpoint': jax_checkpoint_path
        }
    }, output_path)
    
    print(f"PyTorch checkpoint saved to: {output_path}")
    return output_path

def load_normalization_stats(stats_path: str) -> Dict[str, xr.Dataset]:
    """Load normalization statistics from NetCDF files."""
    stats = {}
    
    stats_dir = Path(stats_path)
    if stats_dir.is_dir():
        for stat_file in ['stddev_by_level.nc', 'mean_by_level.nc', 'diffs_stddev_by_level.nc']:
            file_path = stats_dir / stat_file
            if file_path.exists():
                stat_name = stat_file.replace('.nc', '')
                stats[stat_name] = xr.open_dataset(file_path)
                print(f"Loaded {stat_name} from {file_path}")
    else:
        stats['combined'] = xr.open_dataset(stats_path)
    
    return stats

def create_pytorch_model_from_jax_config(jax_checkpoint_path: str) -> graphcast_torch.GraphCast:
    """Create PyTorch model with configuration from JAX checkpoint."""
    
    jax_data = load_jax_checkpoint(jax_checkpoint_path)
    
    model_config_dict = None
    task_config_dict = None
    
    if hasattr(jax_data, 'model_config'):
        model_config_dict = _dataclass_to_dict(jax_data.model_config)
        task_config_dict = _dataclass_to_dict(jax_data.task_config)
    elif 'model_config' in jax_data:
        model_config_dict = _numpy_to_python(jax_data['model_config'])
        task_config_dict = _numpy_to_python(jax_data['task_config'])
    else:
        print("Warning: No model_config found in JAX checkpoint, using GraphCast defaults")
        model_config_dict = {
            'resolution': 0,
            'mesh_size': 6,
            'latent_size': 512,
            'gnn_msg_steps': 16,
            'hidden_layers': 1,
            'radius_query_fraction_edge_length': 0.6
        }
        task_config_dict = {
            'input_variables': ('geopotential', 'temperature', 'u_component_of_wind', 'v_component_of_wind'),
            'target_variables': ('geopotential', 'temperature', 'u_component_of_wind', 'v_component_of_wind'),
            'forcing_variables': ('mean_sea_level_pressure',),
            'pressure_levels': (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
            'input_duration': '12h'
        }
    
    model_config = graphcast_torch.ModelConfig(**model_config_dict)
    task_config = graphcast_torch.TaskConfig(**task_config_dict)
    
    pytorch_model = graphcast_torch.GraphCast(model_config, task_config)
    
    return pytorch_model

def _dataclass_to_dict(dataclass_obj) -> Dict[str, Any]:
    """Convert dataclass to dictionary."""
    if hasattr(dataclass_obj, '__dataclass_fields__'):
        return {f.name: getattr(dataclass_obj, f.name) 
                for f in dataclass_obj.__dataclass_fields__.values()}
    else:
        return dict(dataclass_obj) if hasattr(dataclass_obj, 'items') else dataclass_obj

def _numpy_to_python(obj) -> Any:
    """Convert numpy arrays and types to Python equivalents."""
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:  # scalar
            return obj.item()
        else:
            return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_numpy_to_python(item) for item in obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj

def main():
    """Example usage of JAX to PyTorch conversion."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert JAX GraphCast checkpoint to PyTorch')
    parser.add_argument('--jax_checkpoint', required=True, help='Path to JAX checkpoint file')
    parser.add_argument('--output_path', help='Output path for PyTorch checkpoint')
    parser.add_argument('--normalization_stats', help='Path to normalization statistics')
    
    args = parser.parse_args()
    
    try:
        pytorch_model = create_pytorch_model_from_jax_config(args.jax_checkpoint)
        
        pytorch_checkpoint_path = convert_jax_checkpoint_to_pytorch(
            args.jax_checkpoint, 
            pytorch_model, 
            args.output_path
        )
        
        if args.normalization_stats:
            norm_stats = load_normalization_stats(args.normalization_stats)
            print(f"Loaded normalization statistics: {list(norm_stats.keys())}")
        
        print(f"Conversion completed successfully!")
        print(f"PyTorch model saved to: {pytorch_checkpoint_path}")
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
