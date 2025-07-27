#!/usr/bin/env python3
"""Debug script to check if predictions have gradients."""

import torch
import torch.nn as nn
import xarray as xr
import numpy as np

from graphcast import graphcast_torch
from graphcast import xarray_torch

def create_simple_data():
    """Create minimal synthetic data for debugging."""
    batch_size = 1
    time_steps = 1
    levels = 2
    lat_size = 4
    lon_size = 8
    
    coords = {
        'batch': np.arange(batch_size),
        'time': np.arange(time_steps),
        'level': np.array([500, 850]),
        'lat': np.linspace(-90, 90, lat_size),
        'lon': np.linspace(0, 360, lon_size, endpoint=False)
    }
    
    inputs = xarray_torch.Dataset({
        'geopotential': xarray_torch.DataArray(
            torch.randn(batch_size, time_steps, levels, lat_size, lon_size),
            dims=['batch', 'time', 'level', 'lat', 'lon'],
            coords=coords
        ),
        'temperature': xarray_torch.DataArray(
            torch.randn(batch_size, time_steps, levels, lat_size, lon_size),
            dims=['batch', 'time', 'level', 'lat', 'lon'],
            coords=coords
        )
    })
    
    targets = xarray_torch.Dataset({
        'geopotential': xarray_torch.DataArray(
            torch.randn(batch_size, time_steps, levels, lat_size, lon_size),
            dims=['batch', 'time', 'level', 'lat', 'lon'],
            coords=coords
        ),
        'temperature': xarray_torch.DataArray(
            torch.randn(batch_size, time_steps, levels, lat_size, lon_size),
            dims=['batch', 'time', 'level', 'lat', 'lon'],
            coords=coords
        )
    })
    
    forcings = xarray_torch.Dataset({
        'mean_sea_level_pressure': xarray_torch.DataArray(
            torch.randn(batch_size, time_steps, lat_size, lon_size),
            dims=['batch', 'time', 'lat', 'lon'],
            coords={k: v for k, v in coords.items() if k != 'level'}
        )
    })
    
    return inputs, targets, forcings

def debug_predictions():
    """Debug prediction gradients."""
    print("=== Debugging Prediction Gradients ===")
    
    model_config = graphcast_torch.ModelConfig(
        resolution=0,
        mesh_size=4,
        latent_size=8,
        gnn_msg_steps=1,
        hidden_layers=1,
        radius_query_fraction_edge_length=0.6
    )
    
    task_config = graphcast_torch.TaskConfig(
        input_variables=['geopotential', 'temperature'],
        target_variables=['geopotential', 'temperature'],
        forcing_variables=['mean_sea_level_pressure'],
        pressure_levels=[500, 850],
        input_duration='6h'
    )
    
    model = graphcast_torch.GraphCast(model_config, task_config)
    inputs, targets, forcings = create_simple_data()
    
    print("Initializing model...")
    with torch.no_grad():
        _ = model(inputs, targets, forcings)
    
    total_params = sum(p.numel() for p in model.parameters())
    grad_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Model parameters: {total_params:,}")
    print(f"Parameters requiring grad: {len(grad_params)}")
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter {name}: shape={param.shape}, requires_grad={param.requires_grad}")
            break
    
    print("Forward pass...")
    model.train()
    predictions = model(inputs, targets, forcings)
    
    print("Checking predictions...")
    for var_name in task_config.target_variables:
        if var_name in predictions.data_vars:
            pred_data = xarray_torch.torch_data(predictions[var_name])
            print(f"Prediction {var_name}:")
            print(f"  Shape: {pred_data.shape}")
            print(f"  Requires grad: {pred_data.requires_grad}")
            print(f"  Grad fn: {pred_data.grad_fn}")
            
            if pred_data.requires_grad:
                print(f"  ✓ Prediction has gradients!")
            else:
                print(f"  ❌ Prediction doesn't have gradients")
    
    print("Manual loss computation...")
    pred_geo = xarray_torch.torch_data(predictions['geopotential'])
    target_geo = xarray_torch.torch_data(targets['geopotential'])
    
    print(f"Pred geo requires_grad: {pred_geo.requires_grad}")
    print(f"Target geo requires_grad: {target_geo.requires_grad}")
    
    manual_loss = torch.nn.functional.mse_loss(pred_geo, target_geo)
    print(f"Manual loss requires_grad: {manual_loss.requires_grad}")
    print(f"Manual loss grad_fn: {manual_loss.grad_fn}")
    
    if manual_loss.requires_grad:
        print("✓ Manual loss has gradients!")
        try:
            manual_loss.backward()
            print("✓ Manual backward pass successful!")
            
            grad_count = 0
            for param in model.parameters():
                if param.grad is not None:
                    grad_count += 1
            print(f"Parameters with gradients: {grad_count}")
            
        except Exception as e:
            print(f"❌ Manual backward pass failed: {e}")
    else:
        print("❌ Manual loss doesn't have gradients")

if __name__ == "__main__":
    debug_predictions()
