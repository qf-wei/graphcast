#!/usr/bin/env python3
"""Debug script to isolate gradient computation issues."""

import torch
import torch.nn as nn
import xarray as xr
import numpy as np

from graphcast import graphcast_torch
from graphcast import xarray_torch
from graphcast.normalization_torch import InputsAndResiduals
from graphcast.autoregressive_torch import Predictor as AutoregressivePredictor

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

def debug_base_model():
    """Debug the base GraphCast model loss computation."""
    print("=== Debugging Base GraphCast Model ===")
    
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
    
    base_model = graphcast_torch.GraphCast(model_config, task_config)
    
    inputs, targets, forcings = create_simple_data()
    
    print("Initializing base model...")
    with torch.no_grad():
        _ = base_model(inputs, targets, forcings)
    
    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"Base model parameters: {total_params:,}")
    
    grad_params = [p for p in base_model.parameters() if p.requires_grad]
    print(f"Parameters requiring grad: {len(grad_params)}")
    
    print("Computing loss...")
    base_model.train()
    loss, diagnostics = base_model.loss(inputs, targets, forcings)
    
    print(f"Loss type: {type(loss)}")
    print(f"Loss value: {loss}")
    print(f"Loss requires_grad: {loss.requires_grad}")
    print(f"Loss grad_fn: {loss.grad_fn}")
    
    if loss.requires_grad:
        print("✓ Base model loss has gradients!")
        try:
            loss.backward()
            print("✓ Backward pass successful!")
        except Exception as e:
            print(f"❌ Backward pass failed: {e}")
    else:
        print("❌ Base model loss doesn't require gradients")
    
    return base_model, inputs, targets, forcings

def debug_wrapped_model():
    """Debug the wrapped model (with normalization and autoregressive)."""
    print("\n=== Debugging Wrapped Model ===")
    
    base_model, inputs, targets, forcings = debug_base_model()
    
    print("Creating normalization stats...")
    stddev_by_level = xr.Dataset({
        'geopotential': xr.DataArray([1.0, 1.0], dims=['level'], coords={'level': [500, 850]}),
        'temperature': xr.DataArray([1.0, 1.0], dims=['level'], coords={'level': [500, 850]}),
        'mean_sea_level_pressure': xr.DataArray(1.0)
    })
    
    mean_by_level = xr.Dataset({
        'geopotential': xr.DataArray([0.0, 0.0], dims=['level'], coords={'level': [500, 850]}),
        'temperature': xr.DataArray([0.0, 0.0], dims=['level'], coords={'level': [500, 850]}),
        'mean_sea_level_pressure': xr.DataArray(0.0)
    })
    
    diffs_stddev_by_level = stddev_by_level.copy()
    
    print("Wrapping with normalization...")
    normalized_model = InputsAndResiduals(
        base_model,
        stddev_by_level=stddev_by_level,
        mean_by_level=mean_by_level,
        diffs_stddev_by_level=diffs_stddev_by_level
    )
    
    print("Testing normalization wrapper...")
    normalized_model.train()
    loss, diagnostics = normalized_model.loss(inputs, targets, forcings)
    
    print(f"Normalized loss type: {type(loss)}")
    print(f"Normalized loss value: {loss}")
    print(f"Normalized loss requires_grad: {loss.requires_grad}")
    print(f"Normalized loss grad_fn: {loss.grad_fn}")
    
    if loss.requires_grad:
        print("✓ Normalized model loss has gradients!")
        try:
            loss.backward()
            print("✓ Normalized backward pass successful!")
        except Exception as e:
            print(f"❌ Normalized backward pass failed: {e}")
    else:
        print("❌ Normalized model loss doesn't require gradients")
    
    print("Wrapping with autoregressive...")
    autoregressive_model = AutoregressivePredictor(normalized_model)
    
    print("Testing autoregressive wrapper...")
    autoregressive_model.train()
    loss, diagnostics = autoregressive_model.loss(inputs, targets, forcings)
    
    print(f"Autoregressive loss type: {type(loss)}")
    print(f"Autoregressive loss value: {loss}")
    print(f"Autoregressive loss requires_grad: {loss.requires_grad}")
    print(f"Autoregressive loss grad_fn: {loss.grad_fn}")
    
    if loss.requires_grad:
        print("✓ Autoregressive model loss has gradients!")
        try:
            loss.backward()
            print("✓ Autoregressive backward pass successful!")
        except Exception as e:
            print(f"❌ Autoregressive backward pass failed: {e}")
    else:
        print("❌ Autoregressive model loss doesn't require gradients")

if __name__ == "__main__":
    debug_wrapped_model()
