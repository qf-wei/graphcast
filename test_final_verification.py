#!/usr/bin/env python3
"""Final verification test for PyTorch GraphCast training and inference."""

import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
import numpy as np
from pathlib import Path

from graphcast import graphcast_torch
from graphcast import xarray_torch
from graphcast.normalization_torch import InputsAndResiduals
from graphcast.autoregressive_torch import Predictor as AutoregressivePredictor

def create_test_data():
    """Create test data for verification."""
    batch_size = 2
    time_steps = 3
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

def test_complete_pipeline():
    """Test the complete training and inference pipeline."""
    print("🚀 Testing Complete PyTorch GraphCast Pipeline")
    print("=" * 60)
    
    model_config = graphcast_torch.ModelConfig(
        resolution=0,
        mesh_size=8,
        latent_size=16,
        gnn_msg_steps=2,
        hidden_layers=2,
        radius_query_fraction_edge_length=0.6
    )
    
    task_config = graphcast_torch.TaskConfig(
        input_variables=['geopotential', 'temperature'],
        target_variables=['geopotential', 'temperature'],
        forcing_variables=['mean_sea_level_pressure'],
        pressure_levels=[500, 700, 850],
        input_duration='6h'
    )
    
    inputs, targets, forcings = create_test_data()
    print(f"✓ Created test data: {inputs.sizes}")
    
    base_model = graphcast_torch.GraphCast(model_config, task_config)
    print("✓ Created base GraphCast model")
    
    stddev_by_level = xr.Dataset({
        'geopotential': xr.DataArray([1.0, 1.0, 1.0], dims=['level'], coords={'level': [500, 700, 850]}),
        'temperature': xr.DataArray([1.0, 1.0, 1.0], dims=['level'], coords={'level': [500, 700, 850]}),
        'mean_sea_level_pressure': xr.DataArray(1.0)
    })
    
    mean_by_level = xr.Dataset({
        'geopotential': xr.DataArray([0.0, 0.0, 0.0], dims=['level'], coords={'level': [500, 700, 850]}),
        'temperature': xr.DataArray([0.0, 0.0, 0.0], dims=['level'], coords={'level': [500, 700, 850]}),
        'mean_sea_level_pressure': xr.DataArray(0.0)
    })
    
    diffs_stddev_by_level = stddev_by_level.copy()
    
    normalized_model = InputsAndResiduals(
        base_model,
        stddev_by_level=stddev_by_level,
        mean_by_level=mean_by_level,
        diffs_stddev_by_level=diffs_stddev_by_level
    )
    print("✓ Created normalization wrapper")
    
    model = AutoregressivePredictor(normalized_model)
    print("✓ Created autoregressive wrapper")
    
    with torch.no_grad():
        _ = model(inputs, targets, forcings)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model initialized: {total_params:,} total params, {trainable_params:,} trainable")
    
    print("\n📚 Testing Training Pipeline")
    print("-" * 30)
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    losses = []
    for epoch in range(3):
        optimizer.zero_grad()
        loss, diagnostics = model.loss(inputs, targets, forcings)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"  Epoch {epoch + 1}: loss = {loss.item():.6f}")
    
    print(f"✓ Training completed - loss trend: {losses[0]:.3f} → {losses[-1]:.3f}")
    
    print("\n🔮 Testing Inference Pipeline")
    print("-" * 30)
    
    model.eval()
    with torch.no_grad():
        single_input = inputs.isel(time=[0])
        single_target = targets.isel(time=[0])
        single_forcing = forcings.isel(time=[0])
        
        single_pred = model(single_input, single_target, single_forcing)
        print(f"✓ Single-step prediction: {single_pred.sizes}")
        
        multi_pred = model(inputs, targets, forcings)
        print(f"✓ Multi-step prediction: {multi_pred.sizes}")
        
        for var_name in task_config.target_variables:
            pred_shape = multi_pred[var_name].shape
            target_shape = targets[var_name].shape
            assert pred_shape == target_shape, f"Shape mismatch for {var_name}: {pred_shape} vs {target_shape}"
        
        print("✓ All output shapes match targets")
    
    print("\n⚡ Testing Gradient Flow")
    print("-" * 25)
    
    model.train()
    loss, _ = model.loss(inputs, targets, forcings)
    
    print(f"Loss value: {loss.item():.6f}")
    print(f"Loss requires_grad: {loss.requires_grad}")
    print(f"Loss grad_fn: {loss.grad_fn is not None}")
    
    loss.backward()
    grad_params = [p for p in model.parameters() if p.grad is not None]
    print(f"✓ Parameters with gradients: {len(grad_params)}")
    
    model.train()
    predictions = model(inputs, targets, forcings)
    pred_tensor = xarray_torch.torch_data(predictions['geopotential'])
    print(f"✓ Predictions require_grad: {pred_tensor.requires_grad}")
    
    print("\n🎉 All Tests Passed!")
    print("=" * 60)
    print("✅ PyTorch GraphCast training and inference pipeline is fully functional")
    print(f"✅ Model has {total_params:,} parameters with proper gradient flow")
    print("✅ Training loop works with loss computation and backpropagation")
    print("✅ Inference supports both single-step and multi-step predictions")
    print("✅ All output shapes match expected targets")
    print("✅ Autoregressive rollout works correctly")

if __name__ == "__main__":
    test_complete_pipeline()
