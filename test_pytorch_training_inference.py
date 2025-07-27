#!/usr/bin/env python3
"""Comprehensive test script for PyTorch training and inference."""

import torch
import torch.nn as nn
import xarray as xr
import numpy as np
import tempfile
from pathlib import Path
import time

from graphcast import graphcast_torch
from graphcast import xarray_torch
from graphcast.normalization_torch import InputsAndResiduals
from graphcast.autoregressive_torch import Predictor as AutoregressivePredictor


def create_synthetic_data(task_config: graphcast_torch.TaskConfig, 
                         num_target_steps: int = 1) -> tuple:
    """Create synthetic weather data for testing."""
    batch_size = 1
    time_steps_input = 2
    lat_size = 16
    lon_size = 32
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
        
        data = torch.randn(*shape, dtype=torch.float32)
        inputs[var] = xarray_torch.DataArray(data, dims=dims, coords={k: coords[k] for k in dims})
    
    targets = {}
    for var in task_config.target_variables:
        if var in ['geopotential', 'temperature', 'u_component_of_wind', 'v_component_of_wind']:
            shape = (batch_size, num_target_steps, len(levels), lat_size, lon_size)
            dims = ['batch', 'time', 'level', 'lat', 'lon']
        else:
            shape = (batch_size, num_target_steps, lat_size, lon_size)
            dims = ['batch', 'time', 'lat', 'lon']
        
        data = torch.randn(*shape, dtype=torch.float32)
        targets[var] = xarray_torch.DataArray(data, dims=dims, coords={k: target_coords[k] for k in dims})
    
    forcings = {}
    for var in task_config.forcing_variables:
        shape = (batch_size, num_target_steps, lat_size, lon_size)
        dims = ['batch', 'time', 'lat', 'lon']
        data = torch.randn(*shape, dtype=torch.float32)
        forcings[var] = xarray_torch.DataArray(data, dims=dims, coords={k: target_coords[k] for k in dims})
    
    return xarray_torch.Dataset(inputs), xarray_torch.Dataset(targets), xarray_torch.Dataset(forcings)


def create_synthetic_normalization_stats() -> tuple:
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


def test_model_creation():
    """Test creating the complete model."""
    print("Testing model creation...")
    
    model_config = graphcast_torch.ModelConfig(
        resolution=1,
        mesh_size=2,
        latent_size=32,
        gnn_msg_steps=2,
        hidden_layers=1,
        radius_query_fraction_edge_length=1.0
    )
    
    task_config = graphcast_torch.TaskConfig(
        input_variables=('geopotential', 'temperature'),
        target_variables=('geopotential', 'temperature'),
        forcing_variables=('2m_temperature',),
        pressure_levels=(500, 850),
        input_duration='12h'
    )
    
    base_model = graphcast_torch.GraphCast(model_config, task_config)
    
    stddev_by_level, mean_by_level, diffs_stddev_by_level = create_synthetic_normalization_stats()
    normalized_model = InputsAndResiduals(
        base_model,
        stddev_by_level=stddev_by_level,
        mean_by_level=mean_by_level,
        diffs_stddev_by_level=diffs_stddev_by_level
    )
    
    model = AutoregressivePredictor(normalized_model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created successfully ({total_params:,} parameters)")
    if total_params == 0:
        print("WARNING: Model has 0 parameters - LazyLinear layers not yet initialized")
    
    return model, task_config


def test_forward_pass():
    """Test forward pass."""
    print("Testing forward pass...")
    
    model, task_config = test_model_creation()
    model.eval()
    
    inputs, targets, forcings = create_synthetic_data(task_config, num_target_steps=1)
    
    with torch.no_grad():
        predictions = model(inputs, targets, forcings)
    
    assert isinstance(predictions, xr.Dataset)
    assert len(predictions.data_vars) > 0
    
    for var_name in task_config.target_variables:
        assert var_name in predictions.data_vars
        pred_var = predictions[var_name]
        target_var = targets[var_name]
        print(f"Variable {var_name}:")
        print(f"  Prediction dims: {pred_var.dims}, shape: {pred_var.shape}")
        print(f"  Target dims: {target_var.dims}, shape: {target_var.shape}")
        assert pred_var.dims == target_var.dims, f"Dimension mismatch for {var_name}: pred {pred_var.dims} vs target {target_var.dims}"
    
    print("✓ Forward pass test passed")
    return model, task_config


def test_loss_computation():
    """Test loss computation."""
    print("Testing loss computation...")
    
    model, task_config = test_forward_pass()
    
    inputs, targets, forcings = create_synthetic_data(task_config, num_target_steps=1)
    
    loss, diagnostics = model.loss(inputs, targets, forcings)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0
    assert isinstance(diagnostics, xr.Dataset)
    
    print(f"✓ Loss computation test passed (loss: {loss.item():.6f})")
    return model, task_config


def test_gradient_computation():
    """Test gradient computation and backpropagation."""
    print("Testing gradient computation...")
    
    model, task_config = test_loss_computation()
    model.train()
    
    inputs, targets, forcings = create_synthetic_data(task_config, num_target_steps=1)
    
    print("Initializing model parameters with forward pass...")
    with torch.no_grad():
        _ = model(inputs, targets, forcings)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {total_params:,} parameters")
    
    if total_params == 0:
        print("WARNING: Model still has 0 parameters after initialization")
        return model, task_config
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    initial_params = [p.clone() for p in model.parameters()]
    
    optimizer.zero_grad()
    loss, diagnostics = model.loss(inputs, targets, forcings)
    loss.backward()
    optimizer.step()
    
    params_changed = False
    for initial_p, current_p in zip(initial_params, model.parameters()):
        if not torch.allclose(initial_p, current_p, atol=1e-6):
            params_changed = True
            break
    
    assert params_changed, "Parameters should change after gradient update"
    print(f"✓ Gradient computation test passed (loss: {loss.item():.6f})")
    return model, task_config


def test_autoregressive_prediction():
    """Test autoregressive prediction."""
    print("Testing autoregressive prediction...")
    
    model, task_config = test_gradient_computation()
    model.eval()
    
    inputs, targets, forcings = create_synthetic_data(task_config, num_target_steps=3)
    
    with torch.no_grad():
        predictions = model(inputs, targets, forcings)
    
    assert isinstance(predictions, xr.Dataset)
    assert predictions.dims['time'] == targets.dims['time']
    
    for var_name in task_config.target_variables:
        assert var_name in predictions.data_vars
        pred_var = predictions[var_name]
        target_var = targets[var_name]
        assert pred_var.shape == target_var.shape
    
    print("✓ Autoregressive prediction test passed")
    return model, task_config


def test_training_loop():
    """Test complete training loop."""
    print("Testing training loop...")
    
    model, task_config = test_autoregressive_prediction()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    losses = []
    for epoch in range(3):
        model.train()
        
        inputs, targets, forcings = create_synthetic_data(task_config, num_target_steps=1)
        
        optimizer.zero_grad()
        loss, diagnostics = model.loss(inputs, targets, forcings)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f"  Epoch {epoch + 1}: loss = {loss.item():.6f}")
    
    print("✓ Training loop test passed")
    return model, task_config, losses


def test_inference_modes():
    """Test different inference modes."""
    print("Testing inference modes...")
    
    model, task_config, _ = test_training_loop()
    model.eval()
    
    inputs, targets_template, forcings = create_synthetic_data(task_config, num_target_steps=2)
    
    with torch.no_grad():
        single_step = model(inputs, targets_template.isel(time=[0]), forcings.isel(time=[0]))
        multi_step = model(inputs, targets_template, forcings)
    
    assert isinstance(single_step, xr.Dataset)
    assert isinstance(multi_step, xr.Dataset)
    assert single_step.dims['time'] == 1
    assert multi_step.dims['time'] == 2
    
    print("✓ Inference modes test passed")


def test_checkpoint_save_load():
    """Test checkpoint saving and loading."""
    print("Testing checkpoint save/load...")
    
    model, task_config, _ = test_training_loop()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir) / "test_checkpoint.pt"
        
        model_config = graphcast_torch.ModelConfig(
            resolution=1,
            mesh_size=2,
            latent_size=32,
            gnn_msg_steps=2,
            hidden_layers=1,
            radius_query_fraction_edge_length=1.0
        )
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': model_config,
            'task_config': task_config,
        }
        torch.save(checkpoint, checkpoint_path)
        
        loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        new_model_config = loaded_checkpoint['model_config']
        new_task_config = loaded_checkpoint['task_config']
        
        new_base_model = graphcast_torch.GraphCast(new_model_config, new_task_config)
        stddev_by_level, mean_by_level, diffs_stddev_by_level = create_synthetic_normalization_stats()
        new_normalized_model = InputsAndResiduals(
            new_base_model,
            stddev_by_level=stddev_by_level,
            mean_by_level=mean_by_level,
            diffs_stddev_by_level=diffs_stddev_by_level
        )
        new_model = AutoregressivePredictor(new_normalized_model)
        new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
        
        inputs, targets, forcings = create_synthetic_data(task_config, num_target_steps=1)
        
        with torch.no_grad():
            original_pred = model(inputs, targets, forcings)
            loaded_pred = new_model(inputs, targets, forcings)
        
        for var_name in task_config.target_variables:
            orig_data = xarray_torch.torch_data(original_pred[var_name])
            loaded_data = xarray_torch.torch_data(loaded_pred[var_name])
            assert torch.allclose(orig_data, loaded_data, atol=1e-5)
    
    print("✓ Checkpoint save/load test passed")


def main():
    """Run all tests."""
    print("Running comprehensive PyTorch training and inference tests...")
    print("=" * 60)
    
    try:
        test_model_creation()
        test_forward_pass()
        test_loss_computation()
        test_gradient_computation()
        test_autoregressive_prediction()
        test_training_loop()
        test_inference_modes()
        test_checkpoint_save_load()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("PyTorch training and inference implementation is working correctly!")
        return 0
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
